"""Plot super-resolution inference output against the ClimBench target.

Reads the merged NetCDF written by ``merge_nc_shards.py`` (or any file with
``<variable>(time, pix)`` and a HEALPix ``crs``), looks up the same days in
``<data-root>/experiment<N>/healpix/target_<split>.zarr`` and writes

  * ``<out-dir>/<variable>_<date>.png``: generated / target / error, global + zoom,
  * ``<out-dir>/<variable>_{rmse,rmse_interp,bias}_map.png``: error statistics over
    all days (or every ``--stats-stride``-th day) at every pixel, where "interp" is
    the baseline of average-pooling the target to ``--lr-level`` and interpolating,
  * ``_seasonal_rmse.png`` and ``_zonal_rmse.png``: the same RMSE by season / latitude,
  * ``<out-dir>/<variable>_metrics.txt``: global RMSE / bias per day and per season.

CPU only. Usage:
    python scripts/climbench_sr_plot.py out/exp1_step180000_test_merged.nc \
        --experiment 1 --out-dir plots/exp1 --num-days 4
    python scripts/climbench_sr_plot.py ... --dates 2025-07-01 2025-12-31
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch
import earth2grid
import xarray as xr
from earth2grid import healpix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climbench_sr import DATA_ROOT

NLAT, NLON = 360, 720
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
ORDERS = {"ring": healpix.PixelOrder.RING, "nest": healpix.PixelOrder.NEST}


class LatLonView:
    """HEALPix (RING) -> regular lat/lon image on -180..180, for plotting only."""

    def __init__(self, level, zoom_box):
        grid = healpix.Grid(level=level, pixel_order=healpix.PixelOrder.RING)
        lat = np.linspace(-90, 90, NLAT)
        lon360 = np.linspace(0, 360, NLON, endpoint=False)
        self.regrid = grid.get_bilinear_regridder_to(
            lat[:, None], lon360[None, :]
        ).float()
        lon = ((lon360 + 180) % 360) - 180
        self.col_order = np.argsort(lon)
        lon = lon[self.col_order]
        lon_w, lon_e, lat_s, lat_n = zoom_box
        self.zoom_box = zoom_box
        self.zoom = (
            slice(int(np.searchsorted(lat, lat_s)), int(np.searchsorted(lat, lat_n))),
            slice(int(np.searchsorted(lon, lon_w)), int(np.searchsorted(lon, lon_e))),
        )

    def __call__(self, x):  # (npix,) -> (nlat, nlon)
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return self.regrid(x[None])[0].numpy()[:, self.col_order]


def to_ring(x, order):
    if order == healpix.PixelOrder.RING:
        return np.asarray(x)
    return healpix.reorder(
        torch.as_tensor(np.asarray(x)), order, healpix.PixelOrder.RING
    ).numpy()


def show(ax, view, img, box_name, **kw):
    box = (-180, 180, -90, 90) if box_name == "global" else view.zoom_box
    sl = (slice(None), slice(None)) if box_name == "global" else view.zoom
    im = ax.imshow(
        img[sl], origin="lower", extent=box, aspect="auto", **kw
    )
    return im


def day_figure(view, var, date, pred, target):
    p, t = view(pred), view(target)
    err = p - t
    vmin, vmax = np.percentile(t, [1, 99])
    elim = np.percentile(np.abs(err), 99)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
    for i, box in enumerate(["global", "zoom"]):
        for j, (img, title) in enumerate(
            [(p, "generated"), (t, "target"), (err, "generated - target")]
        ):
            if j < 2:
                kw = dict(vmin=vmin, vmax=vmax, cmap="viridis")
            else:
                kw = dict(vmin=-elim, vmax=elim, cmap="RdBu_r")
            im = show(axes[i, j], view, img, box, **kw)
            if i == 0:
                rmse = np.sqrt(np.mean(err**2))
                axes[i, j].set_title(
                    f"{title} (rmse {rmse:.3g})" if j == 2 else title
                )
            if i == 1:
                fig.colorbar(im, ax=axes[:, j], shrink=0.7, location="bottom")
    fig.suptitle(f"{var}  {date}")
    return fig


def map_figure(view, var, title, img, cmap, symmetric):
    lim = np.percentile(np.abs(img), 99) if symmetric else None
    kw = dict(vmin=-lim, vmax=lim) if symmetric else dict(vmin=0, vmax=np.percentile(img, 99))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for ax, box in zip(axes, ["global", "zoom"]):
        im = show(ax, view, img, box, cmap=cmap, **kw)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"{var}  {title}")
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prediction", help="merged NetCDF with <variable>(time, pix)")
    p.add_argument("--experiment", type=int, required=True)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--split", default="test", choices=["test", "validation", "train"])
    p.add_argument("--variable", default=None, help="default: the only data variable")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dates", nargs="+", default=None, help="YYYY-MM-DD")
    p.add_argument("--num-days", type=int, default=4, help="evenly spaced days")
    p.add_argument(
        "--stats-stride",
        type=int,
        default=1,
        help="use every N-th day for the RMSE/bias maps (0 skips them)",
    )
    p.add_argument("--lr-level", type=int, default=3, help="input level of the baseline")
    p.add_argument(
        "--zoom-box",
        type=float,
        nargs=4,
        default=[-20, 50, 30, 70],
        metavar=("LON_W", "LON_E", "LAT_S", "LAT_N"),
    )
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = nc.Dataset(args.prediction)
    var = args.variable or [
        v for v in ds.variables if ds[v].dimensions[:1] == ("time",) and v != "time"
    ][0]
    order = ORDERS[ds[ds[var].grid_mapping].healpix_order]
    level = healpix.npix2level(ds.dimensions["pix"].size)
    pred_time = np.asarray(ds["time"][:]).astype("datetime64[s]")

    path = os.path.join(
        args.data_root, f"experiment{args.experiment}", "healpix", f"target_{args.split}.zarr"
    )
    zt = xr.open_zarr(path, chunks=None)
    assert zt.attrs.get("healpix_order", "nested") == "nested"
    target_order = healpix.PixelOrder.NEST
    tgt_time = zt["time"].values.astype("datetime64[s]")
    # row of the target for each prediction time
    lookup = {t: i for i, t in enumerate(tgt_time)}
    missing = [t for t in pred_time if t not in lookup]
    if missing:
        raise SystemExit(f"{len(missing)} prediction days not in {path}, e.g. {missing[0]}")
    tgt_idx = np.array([lookup[t] for t in pred_time])

    def get(k):
        pred = to_ring(ds[var][k], order)
        target = to_ring(zt[var].isel(time=int(tgt_idx[k])).values, target_order)
        return pred, target

    view = LatLonView(level, args.zoom_box)

    if args.dates:
        days = [int(np.flatnonzero(pred_time == np.datetime64(d, "s"))[0]) for d in args.dates]
    else:
        days = np.linspace(0, len(pred_time) - 1, args.num_days).round().astype(int)
    for k in days:
        date = str(pred_time[k])[:10]
        fig = day_figure(view, var, date, *get(k))
        fig.savefig(os.path.join(args.out_dir, f"{var}_{date}.png"), dpi=110)
        plt.close(fig)
        print("wrote", date, flush=True)

    if args.stats_stride > 0:
        ks = range(0, len(pred_time), args.stats_stride)
        npix = ds.dimensions["pix"].size
        # interpolation baseline: average-pool the target to the low-res level and
        # interpolate back (what the model receives as input), all in NEST order
        lr_level = args.lr_level
        regrid = earth2grid.get_regridder(
            healpix.Grid(level=lr_level, pixel_order=healpix.PixelOrder.NEST),
            healpix.Grid(level=level, pixel_order=healpix.PixelOrder.NEST),
        ).float()

        def baseline(target_ring):
            x = torch.as_tensor(target_ring, dtype=torch.float32)
            x = healpix.reorder(x, healpix.PixelOrder.RING, healpix.PixelOrder.NEST)
            for _ in range(level - lr_level):
                x = x.reshape(-1, 4).mean(-1)
            y = regrid(x)
            return healpix.reorder(y, healpix.PixelOrder.NEST, healpix.PixelOrder.RING).numpy()

        months = pred_time.astype("datetime64[M]").astype(int) % 12 + 1
        season_of = {m: s for s, ms in SEASONS.items() for m in ms}
        sq = {s: np.zeros(npix) for s in [*SEASONS, "all"]}
        sq_base = {s: np.zeros(npix) for s in sq}
        count = dict.fromkeys(sq, 0)
        bias = np.zeros(npix)
        lines = ["date rmse rmse_interp bias"]
        for n, k in enumerate(ks):
            pred, target = get(k)
            e = pred.astype(np.float64) - target
            eb = baseline(target).astype(np.float64) - target
            for key in ("all", season_of[int(months[k])]):
                sq[key] += e**2
                sq_base[key] += eb**2
                count[key] += 1
            bias += e
            lines.append(
                f"{str(pred_time[k])[:10]} {np.sqrt((e**2).mean()):.5g} "
                f"{np.sqrt((eb**2).mean()):.5g} {e.mean():.5g}"
            )
            if n % 200 == 0:
                print(f"stats {n + 1}/{len(ks)}", flush=True)
        n_all = count["all"]
        rmse = {s: np.sqrt(sq[s] / count[s]) for s in sq if count[s]}
        rmse_base = {s: np.sqrt(sq_base[s] / count[s]) for s in sq if count[s]}
        bias /= n_all

        for name, img, cmap, sym in [
            ("rmse_map", rmse["all"], "magma", False),
            ("rmse_interp_map", rmse_base["all"], "magma", False),
            ("bias_map", bias, "RdBu_r", True),
        ]:
            fig = map_figure(view, var, f"{name} ({n_all} days)", view(img), cmap, sym)
            fig.savefig(os.path.join(args.out_dir, f"{var}_{name}.png"), dpi=110)
            plt.close(fig)

        # seasonal RMSE maps, model on top and interpolation baseline below
        seasons = [s for s in SEASONS if count[s]]
        vmax = np.percentile(rmse_base["all"], 99)
        fig, axes = plt.subplots(2, len(seasons), figsize=(5 * len(seasons), 6), constrained_layout=True)
        axes = np.atleast_2d(axes).reshape(2, -1)
        for j, s in enumerate(seasons):
            for i, (r, lab) in enumerate([(rmse, "model"), (rmse_base, "interp")]):
                im = show(axes[i, j], view, view(r[s]), "global", cmap="magma", vmin=0, vmax=vmax)
                axes[i, j].set_title(f"{s} {lab} (rmse {np.sqrt((r[s]**2).mean()):.3g})")
        fig.colorbar(im, ax=axes, shrink=0.7, location="right")
        fig.suptitle(f"{var}  seasonal rmse")
        fig.savefig(os.path.join(args.out_dir, f"{var}_seasonal_rmse.png"), dpi=110)
        plt.close(fig)

        # zonal mean RMSE (HEALPix is equal-area, so a latitude-bin mean is unweighted)
        lat = healpix.Grid(level=level, pixel_order=healpix.PixelOrder.RING).lat
        bins = np.linspace(-90, 90, 37)
        idx = np.digitize(lat, bins) - 1
        centers = 0.5 * (bins[1:] + bins[:-1])
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for r, lab in [(rmse, "model"), (rmse_base, "interpolation")]:
            ax.plot(centers, [np.sqrt((r["all"][idx == b] ** 2).mean()) for b in range(36)], label=lab)
        ax.set_xlabel("latitude")
        ax.set_ylabel("rmse")
        ax.legend()
        fig.savefig(os.path.join(args.out_dir, f"{var}_zonal_rmse.png"), dpi=110)
        plt.close(fig)

        lines.append("")
        lines.append("season n_days rmse rmse_interp")
        for s in [*seasons, "all"]:
            lines.append(
                f"{s} {count[s]} {np.sqrt((rmse[s]**2).mean()):.5g} "
                f"{np.sqrt((rmse_base[s]**2).mean()):.5g}"
            )
        with open(os.path.join(args.out_dir, f"{var}_metrics.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines[-(len(seasons) + 2):]))
        print(f"mean bias {bias.mean():.4g}")


if __name__ == "__main__":
    main()
