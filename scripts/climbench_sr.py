"""Train / run the cBottle super-resolution model on ClimBench HEALPix data.

Expected layout (per experiment):

    <data-root>/experiment<N>/healpix/{target,input}_{train,validation,test}.zarr

Only the ``target_*`` stores are read. The low-resolution input is derived from the
target by average pooling inside the training loop / inference script, which is
identical to ``input_*.zarr`` (conservative coarsening on nested HEALPix).

Usage:
    python scripts/climbench_sr.py train --experiment 1 --output-path runs/exp1
    python scripts/climbench_sr.py infer --experiment 1 runs/exp1/cBottle-SR-<step>.zip out/exp1
"""

import argparse
import functools
import json
import os
import random
import sys

import earth2grid
import numpy as np
import torch
import cbottle.distributed as dist
import xarray as xr

from cbottle.datasets import samplers
from cbottle.datasets.base import BatchInfo, TimeUnit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = "/work/bk1444/climbench/data"
# TensorBoard events go to <TENSORBOARD_ROOT>/<run name>/{train,eval}
TENSORBOARD_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tensorboard")
)

# Precomputed per-variable mean / std (scale) of the training targets.
STATS: dict[int, dict[str, tuple[float, float]]] = {
    1: {"tas": (286.99915542783003, 15.414200571501576)},
    2: {"tp": (0.0028682559418698973, 0.006175653414683115)},
    3: {
        "msl": (101139.39094338276, 1062.9293798393428),
        "q850": (0.005964962388804392, 0.004028018112375323),
        "tas": (286.99915542783003, 15.414200571501576),
        "tcc": (0.6229747817337858, 0.3009191133842002),
        "tp": (0.0028682559418698973, 0.006175653414683115),
        "u850": (1.1146329207266692, 7.60048045817027),
        "v850": (0.09364891663880362, 5.052339021547254),
        "z500": (55433.52487005346, 2704.6219155593317),
    },
}


class ClimbenchDataset(torch.utils.data.Dataset):
    """Target-only dataset in the format expected by the SR train/inference code.

    Args:
        split: split requested by the caller ("train" or "test").
        split_map: maps the requested split to a zarr suffix, e.g. in training
            ``{"train": "train", "test": "validation"}`` so that the loop's
            periodic validation uses the validation files.
    """

    time_units = "seconds since 1970-1-1 0:0:0"
    calendar = "standard"

    def __init__(
        self,
        split: str,
        *,
        root: str,
        split_map: dict[str, str],
        variables: list[str],
        mean: list[float],
        scale: list[float],
        normalize: bool,
        random_run_length: int | None = None,
        golden_splits: tuple[str, ...] = (),
        stride: int = 1,
    ):
        self.path = os.path.join(root, f"target_{split_map[split]}.zarr")
        self.normalize = normalize
        ds = xr.open_zarr(self.path, chunks=None)
        self.fields_out = list(variables)
        npix = ds.sizes["cell"]
        self.grid = earth2grid.healpix.Grid(
            level=earth2grid.healpix.npix2level(npix),
            pixel_order=earth2grid.healpix.PixelOrder.NEST,
        )
        assert ds.attrs.get("healpix_order", "nested") == "nested"
        self.times = ds["time"].values.astype("datetime64[s]").astype("int64")
        self.index = np.arange(0, ds.sizes["time"], stride)
        if split in golden_splits:
            # same selection as the repo's default validation: one day per rank,
            # spread over the period with a golden-ratio sequence
            tasks = samplers.subsample(
                range(ds.sizes["time"]), min_samples=dist.get_world_size()
            )
            self.index = np.asarray(samplers.distributed_split(tasks), dtype=int)
        ds.close()

        self.batch_info = BatchInfo(
            self.fields_out,
            scales=[float(x) for x in scale],
            center=[float(x) for x in mean],
            # the checkpoint reader only handles HOUR/MINUTE by name, so a daily
            # step is expressed as 24 hours
            time_step=24,
            time_unit=TimeUnit.HOUR,
        )
        self._mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self._scale = torch.tensor(scale, dtype=torch.float32)[:, None, None]
        self._ds = None  # opened lazily, so the dataset stays picklable
        # state of the random sampling (per process, see _day)
        self.random_run_length = random_run_length
        self._cursor = 0
        self._run_left = 0

    def __len__(self):
        return len(self.index)

    def _day(self, i):
        """Time index for sample ``i``.

        With ``random_run_length`` set this ignores ``i`` and samples like the repo's
        InfiniteSequentialSampler: a random start day, then ``random_run_length``
        consecutive days, then a new random start. The random state is not seeded,
        so every GPU, dataloader worker and restart draws its own sequence.
        """
        if self.random_run_length is None:
            return int(self.index[i])
        if self._run_left == 0:
            self._cursor = random.randrange(len(self.index))
            self._run_left = self.random_run_length
        day = int(self.index[self._cursor])
        self._cursor = (self._cursor + 1) % len(self.index)
        self._run_left -= 1
        return day

    def __getitem__(self, i):
        if self._ds is None:
            self._ds = xr.open_zarr(self.path, chunks=None)
        t = self._day(i)
        x = np.stack([self._ds[v].isel(time=t).values for v in self.fields_out])
        x = torch.from_numpy(x.astype(np.float32))[:, None, :]  # (c, 1, npix)
        if self.normalize:
            x = (x - self._mean) / self._scale
        return {"target": x, "timestamp": float(self.times[t])}


def default_variables(root: str) -> list[str]:
    """Variables present in the train, validation and test targets."""
    names = []
    for split in ("train", "validation", "test"):
        ds = xr.open_zarr(os.path.join(root, f"target_{split}.zarr"), chunks=None)
        names.append([v for v in ds.data_vars if "cell" in ds[v].dims])
    return [v for v in names[0] if all(v in n for n in names[1:])]


def compute_stats(root: str, variables: list[str], n: int = 200):
    """Per-channel mean/std of the training target from a strided subsample."""
    ds = xr.open_zarr(os.path.join(root, "target_train.zarr"), chunks=None)
    idx = np.linspace(0, ds.sizes["time"] - 1, n).astype(int)
    mean, std = [], []
    for v in variables:
        a = ds[v].isel(time=idx).values
        mean.append(float(a.mean()))
        std.append(float(a.std()))
    return mean, std


def get_stats(
    experiment: int, root: str, variables: list[str], cache_dir: str | None
):
    """Precomputed stats if available, otherwise computed (and cached) from data."""
    table = STATS.get(experiment, {})
    if all(v in table for v in variables):
        return [table[v][0] for v in variables], [table[v][1] for v in variables]
    cache = os.path.join(cache_dir, "stats.json") if cache_dir else None
    if cache and os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        if d["variables"] == variables:
            return d["mean"], d["scale"]
        raise ValueError(f"{cache} was computed for {d['variables']}, not {variables}")
    mean, scale = compute_stats(root, variables)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, "w") as f:
            json.dump({"variables": variables, "mean": mean, "scale": scale}, f)
    return mean, scale


def train(args):
    from cbottle.training import super_resolution

    root = os.path.join(args.data_root, f"experiment{args.experiment}", "healpix")
    variables = args.variables or default_variables(root)
    print(f"channels: {variables}", flush=True)
    mean, scale = get_stats(args.experiment, root, variables, args.output_path)
    factory = functools.partial(
        ClimbenchDataset,
        root=root,
        split_map={"train": "train", "test": "validation"},
        variables=variables,
        mean=mean,
        scale=scale,
        normalize=True,
        random_run_length=args.shuffle_every,
        golden_splits=("test",),  # "test" is the validation split during training
    )
    super_resolution.train(
        output_path=args.output_path,
        customized_dataset=factory,
        lr_level=args.lr_level,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        num_steps=args.num_steps,
        log_freq=args.log_freq,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        tensorboard_dir=os.path.join(
            args.tensorboard_root,
            args.run_name or os.path.basename(os.path.normpath(args.output_path)),
            "train",
        ),
    )


def infer(args, rest):
    from inference_multidiffusion import inference

    root = os.path.join(args.data_root, f"experiment{args.experiment}", "healpix")
    # stats saved by training live next to the checkpoint; only used for
    # BatchInfo here, the model normalises with the one stored in the checkpoint
    variables = args.variables or default_variables(root)
    mean, scale = get_stats(
        args.experiment,
        root,
        variables,
        os.path.dirname(os.path.abspath(args.state_path)),
    )
    factory = functools.partial(
        ClimbenchDataset,
        root=root,
        split_map={"test": args.split},
        variables=variables,
        mean=mean,
        scale=scale,
        normalize=False,
        stride=args.stride,
    )
    # the repo default overlap (32) does not tile a level-7 face; 64 matches training
    if "--overlap-size" not in rest:
        rest = ["--overlap-size", "64"] + rest
    inference(
        [args.state_path, args.output_path, "--level", str(args.level)]
        + ["--level-lr", str(args.lr_level)]
        + rest,
        customized_dataset=factory,
    )

    # The comparison with other models only makes sense on the same days: check
    # that exactly one output was written per day of the split. Each rank wrote
    # its own <rank>.nc. (The writer appends to an existing file, so re-using an
    # output folder also fails here.)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # wait until every rank has written its file
    if dist.get_rank() != 0:
        return
    import glob

    import netCDF4 as nc

    expected = len(factory(split="test"))
    times = []
    for path in sorted(glob.glob(os.path.join(args.output_path, "*.nc"))):
        with nc.Dataset(path) as ds:
            times.extend(ds["time"][:].tolist())
    written = len(times)
    if written != expected or len(set(times)) != written:
        raise SystemExit(
            f"wrote {written} time steps ({len(set(times))} distinct) but the "
            f"{args.split} split has {expected} (stride {args.stride}); "
            "use a fresh output folder"
        )
    print(f"wrote {written} time steps, as expected", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--experiment", type=int, required=True)
    parser.add_argument("--lr-level", type=int, default=3, help="HPX level of input")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="channels to use (default: those present in train/validation/test)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--output-path", required=True)
    t.add_argument("--num-steps", type=int, default=200_000)
    t.add_argument("--log-freq", type=int, default=2000, help="validate+checkpoint")
    t.add_argument("--train-batch-size", type=int, default=12)
    t.add_argument("--test-batch-size", type=int, default=12)
    t.add_argument("--dataloader-num-workers", type=int, default=3)
    t.add_argument("--bf16", action="store_true")
    t.add_argument(
        "--run-name",
        default=None,
        help="name under the TensorBoard root (default: basename of --output-path)",
    )
    t.add_argument("--tensorboard-root", default=TENSORBOARD_ROOT)
    t.add_argument(
        "--shuffle-every",
        type=int,
        default=48,
        help="consecutive days per random start, then a new random start "
        "(48 as in the repo's InfiniteSequentialSampler)",
    )

    i = sub.add_parser("infer")
    i.add_argument("state_path")
    i.add_argument("output_path")
    i.add_argument("--level", type=int, default=7, help="HPX level of the target")
    i.add_argument("--split", default="test", choices=["test", "validation", "train"])
    i.add_argument("--stride", type=int, default=1, help="use every N-th time step")

    args, rest = parser.parse_known_args()
    if args.cmd == "train":
        if rest:
            parser.error(f"unrecognized arguments: {rest}")
        train(args)
    else:
        infer(args, rest)


if __name__ == "__main__":
    main()
