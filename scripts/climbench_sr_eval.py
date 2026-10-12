"""Evaluate super-resolution checkpoints and log metrics + maps to TensorBoard.

For every checkpoint ``<run-dir>/cBottle-SR-<step>.zip`` this super-resolves a few
fixed days of a split (default: validation), computes RMSE / bias against the
target and logs global + zoomed maps. Output goes to
``tensorboard/<run-name>/eval`` and the training scalars are in
``tensorboard/<run-name>/train``, so ``tensorboard --logdir tensorboard`` shows
all runs side by side.

Run it on a GPU, next to a training job (``--watch``) or afterwards:
    python scripts/climbench_sr_eval.py --experiment 1 runs/exp1
    python scripts/climbench_sr_eval.py --experiment 1 runs/exp1 --watch
"""

import argparse
import glob
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from earth2grid import healpix
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climbench_sr import (
    DATA_ROOT,
    TENSORBOARD_ROOT,
    ClimbenchDataset,
    default_variables,
    get_stats,
)

from cbottle.inference import Coords, SuperResolutionModel

CKPT_RE = re.compile(r"cBottle-SR-(\d+)\.zip$")
NLAT, NLON = 360, 720


def list_checkpoints(run_dir):
    found = []
    for path in glob.glob(os.path.join(run_dir, "cBottle-SR-*.zip")):
        m = CKPT_RE.search(path)
        # skip files that are still being written
        if m and time.time() - os.path.getmtime(path) > 30:
            found.append((int(m.group(1)), path))
    return sorted(found)


def average_pool(x, levels):
    for _ in range(levels):
        x = x.reshape(x.shape[:-1] + (x.shape[-1] // 4, 4)).mean(-1)
    return x


class LatLonView:
    """HEALPix (nest) -> regular lat/lon image, for plotting only."""

    def __init__(self, level, zoom_box):
        grid = healpix.Grid(level=level, pixel_order=healpix.PixelOrder.NEST)
        self.lat = np.linspace(-90, 90, NLAT)
        lon360 = np.linspace(0, 360, NLON, endpoint=False)
        self.regrid = grid.get_bilinear_regridder_to(
            self.lat[:, None], lon360[None, :]
        ).float()
        # plot on -180..180: reorder the columns
        lon = ((lon360 + 180) % 360) - 180
        self.col_order = np.argsort(lon)
        self.lon = lon[self.col_order]
        lon_w, lon_e, lat_s, lat_n = zoom_box
        self.zoom = (
            slice(
                int(np.searchsorted(self.lat, lat_s)),
                int(np.searchsorted(self.lat, lat_n)),
            ),
            slice(
                int(np.searchsorted(self.lon, lon_w)),
                int(np.searchsorted(self.lon, lon_e)),
            ),
        )
        self.zoom_box = zoom_box

    def __call__(self, x):  # (npix,) -> (nlat, nlon)
        return self.regrid(x.float().cpu()[None])[0].numpy()[:, self.col_order]


def make_figure(view, channel, date, coarse, pred, target):
    images = [view(coarse), view(pred), view(target)]
    titles = [
        "input (interpolated)",
        "generated",
        "target",
    ]
    vmin, vmax = np.percentile(images[2], [1, 99])
    zlat, zlon = view.zoom
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    for j, (img, title) in enumerate(zip(images, titles)):
        rmse = float(np.sqrt(np.mean((img - images[2]) ** 2)))
        for i, (sl, box) in enumerate(
            [((slice(None), slice(None)), (-180, 180, -90, 90)), ((zlat, zlon), view.zoom_box)]
        ):
            ax = axes[i, j]
            im = ax.imshow(
                img[sl],
                origin="lower",
                extent=(box[0], box[1], box[2], box[3]),
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                aspect="auto",
            )
            ax.set_title(f"{title}  (rmse {rmse:.3g})" if i == 0 else title)
        fig.colorbar(im, ax=axes[:, j], shrink=0.7, location="bottom")
    fig.suptitle(f"{channel}  {date}")
    return fig


def evaluate(args, ckpt_step, ckpt_path, dataset, days, view, writer, device):
    t0 = time.time()
    level = dataset.grid.level
    model = SuperResolutionModel.from_pretrained(
        ckpt_path,
        map_location="cpu",
        hpx_level=level,
        hpx_lr_level=args.lr_level,
        patch_size=args.patch_size,
        overlap_size=args.overlap_size,
        num_steps=args.num_steps,
        sigma_max=args.sigma_max,
    ).to(device)
    coords = Coords(dataset.batch_info, model.low_res_grid)
    channels = dataset.batch_info.channels

    sq_err = {c: [] for c in channels}
    sq_err_interp = {c: [] for c in channels}
    bias = {c: [] for c in channels}
    for i in days:
        item = dataset[i]
        target = item["target"][:, 0].to(device)  # (c, npix), physical units
        date = str(np.datetime64(int(item["timestamp"]), "s"))[:10]
        lr = average_pool(target, level - args.lr_level)
        with torch.no_grad():
            pred, _ = model(lr[None, :, None], coords=coords, extents=None)
            interp = model.regrid(lr.float())
        pred = pred[0, :, 0]
        for c, ch in enumerate(channels):
            err = pred[c] - target[c]
            sq_err[ch].append(float((err**2).mean()))
            sq_err_interp[ch].append(float(((interp[c] - target[c]) ** 2).mean()))
            bias[ch].append(float(err.mean()))
            fig = make_figure(view, ch, date, interp[c], pred[c], target[c])
            writer.add_figure(f"{ch}/{date}", fig, ckpt_step)
            plt.close(fig)

    for ch in channels:
        writer.add_scalar(f"eval/{ch}/rmse", np.sqrt(np.mean(sq_err[ch])), ckpt_step)
        writer.add_scalar(
            f"eval/{ch}/rmse_interpolation_baseline",
            np.sqrt(np.mean(sq_err_interp[ch])),
            ckpt_step,
        )
        writer.add_scalar(f"eval/{ch}/bias", np.mean(bias[ch]), ckpt_step)
    writer.flush()
    print(f"step {ckpt_step}: evaluated {len(days)} days in {time.time() - t0:.0f}s")
    del model
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--experiment", type=int, required=True)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--num-days", type=int, default=4, help="evenly spaced days")
    p.add_argument("--lr-level", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument(
        "--overlap-size",
        type=int,
        default=64,
        help="64 matches training and tiles a level-7 face (the repo default 32 does not)",
    )
    p.add_argument("--num-steps", type=int, default=18, help="sampler steps")
    p.add_argument("--sigma-max", type=int, default=800)
    p.add_argument(
        "--zoom-box",
        type=float,
        nargs=4,
        default=(-10, 40, 30, 70),
        metavar=("LON_W", "LON_E", "LAT_S", "LAT_N"),
        help="region for the zoomed maps (default: Europe)",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="name under the TensorBoard root (default: basename of run_dir)",
    )
    p.add_argument("--tensorboard-root", default=TENSORBOARD_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--watch", action="store_true", help="keep waiting for new ckpts")
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--max-idle-minutes", type=int, default=60)
    args = p.parse_args()

    root = os.path.join(args.data_root, f"experiment{args.experiment}", "healpix")
    variables = args.variables or default_variables(root)
    mean, scale = get_stats(args.experiment, root, variables, args.run_dir)
    dataset = ClimbenchDataset(
        "test",
        root=root,
        split_map={"test": args.split},
        variables=variables,
        mean=mean,
        scale=scale,
        normalize=False,
    )
    days = np.linspace(0, len(dataset) - 1, args.num_days).astype(int).tolist()
    view = LatLonView(dataset.grid.level, args.zoom_box)
    run_name = args.run_name or os.path.basename(os.path.normpath(args.run_dir))
    tb_dir = os.path.join(args.tensorboard_root, run_name, "eval")
    writer = SummaryWriter(tb_dir)
    device = torch.device(args.device)

    done_file = os.path.join(tb_dir, "done_steps.txt")
    done = set()
    if os.path.exists(done_file):
        done = {int(x) for x in open(done_file).read().split()}

    idle_since = time.time()
    while True:
        todo = [(s, c) for s, c in list_checkpoints(args.run_dir) if s not in done]
        for step, path in todo:
            try:
                evaluate(args, step, path, dataset, days, view, writer, device)
            except (OSError, RuntimeError) as e:  # e.g. truncated file: retry later
                print(f"step {step}: skipped ({type(e).__name__}: {e})", flush=True)
                continue
            done.add(step)
            with open(done_file, "a") as f:
                f.write(f"{step}\n")
            idle_since = time.time()
        if not args.watch:
            break
        if time.time() - idle_since > args.max_idle_minutes * 60:
            print("no new checkpoints, stopping")
            break
        time.sleep(args.poll_seconds)
    writer.close()


if __name__ == "__main__":
    main()
