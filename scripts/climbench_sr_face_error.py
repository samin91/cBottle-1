"""Does the time-mean error depend on the position inside a HEALPix face?

Accumulates the per-pixel mean squared error of the SR output (and of the
interpolation baseline) in NEST order, folds it onto face coordinates (x, y) in
0..nside-1 and plots profiles / folded maps. Geography is removed by dividing by a
Gaussian-smoothed copy of the same face before averaging ("detrended").
Usage: python scripts/climbench_sr_face_error.py <merged.nc> --experiment 1 --out-dir plots/exp1
"""
import argparse, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch
import xarray as xr
import earth2grid
from earth2grid import healpix
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from climbench_sr import DATA_ROOT
from climbench_sr_plot import to_ring, ORDERS

RING, NEST = healpix.PixelOrder.RING, healpix.PixelOrder.NEST


def deinterleave(v):
    out = np.zeros_like(v)
    for b in range(16):
        out |= ((v >> (2 * b)) & 1) << b
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prediction")
    p.add_argument("--experiment", type=int, required=True)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--variable", default="tas")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--lr-level", type=int, default=3)
    p.add_argument("--smooth", type=float, default=12, help="gaussian sigma (pixels)")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    ds = nc.Dataset(a.prediction)
    var = a.variable
    order = ORDERS[ds[ds[var].grid_mapping].healpix_order]
    npix = ds.dimensions["pix"].size
    level = healpix.npix2level(npix)
    nside = 2**level
    t = np.asarray(ds["time"][:]).astype("datetime64[s]")
    zt = xr.open_zarr(os.path.join(a.data_root, f"experiment{a.experiment}", "healpix",
                                   f"target_{a.split}.zarr"), chunks=None)
    look = {x: i for i, x in enumerate(zt["time"].values.astype("datetime64[s]"))}
    regrid = earth2grid.get_regridder(
        healpix.Grid(level=a.lr_level, pixel_order=NEST), healpix.Grid(level=level, pixel_order=NEST)
    ).float()

    mse = np.zeros(npix); mse_b = np.zeros(npix); n = 0
    for k in range(0, len(t), a.stride):
        pred = to_ring(ds[var][k], order)
        tgt_nest = zt[var].isel(time=look[t[k]]).values.astype(np.float32)
        pred_nest = healpix.reorder(torch.as_tensor(pred), order, NEST).numpy()
        x = torch.as_tensor(tgt_nest)
        for _ in range(level - a.lr_level):
            x = x.reshape(-1, 4).mean(-1)
        base = regrid(x).numpy()
        mse += (pred_nest.astype(np.float64) - tgt_nest) ** 2
        mse_b += (base.astype(np.float64) - tgt_nest) ** 2
        n += 1
    mse /= n; mse_b /= n
    print(f"{n} days")

    pix = np.arange(npix)
    face, within = pix // nside**2, pix % nside**2
    fx, fy = deinterleave(within) , deinterleave(within >> 1)
    # fx uses even bits, fy odd bits (NEST bit interleaving)

    def faces(m):
        g = np.zeros((12, nside, nside)); g[face, fx, fy] = m; return g

    res = {}
    for name, m in [("model", mse), ("interp", mse_b)]:
        g = faces(m)
        sm = np.stack([gaussian_filter(f, a.smooth, mode="nearest") for f in g])
        res[name] = (g, g / sm)
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    for r, name in enumerate(["model", "interp"]):
        g, d = res[name]
        mean_d = d.mean(0)  # average over the 12 faces: (x, y), ratio to local mean
        for ax, m, ttl in [(axes[r, 0], mean_d, "ratio mse / smoothed mse, mean over faces"),
                           (axes[r, 1], d.reshape(12, nside // 64, 64, nside // 64, 64).mean((0, 1, 3)),
                            "folded modulo 64 px (patch stride)")]:
            im = ax.imshow(m.T, origin="lower", cmap="RdBu_r", vmin=0.9, vmax=1.1)
            ax.set_title(f"{name}: {ttl}"); fig.colorbar(im, ax=ax, shrink=0.8)
        ax = axes[r, 2]
        edge = np.minimum.reduce([np.arange(nside)[:, None] + 0 * np.arange(nside)[None],
                                  np.arange(nside)[None] + 0 * np.arange(nside)[:, None],
                                  nside - 1 - np.arange(nside)[:, None] + 0 * np.arange(nside)[None],
                                  nside - 1 - np.arange(nside)[None] + 0 * np.arange(nside)[:, None]])
        prof = [d[:, edge == e].mean() for e in range(nside // 2)]
        ax.plot(prof); ax.axhline(1, c="k", lw=.5); ax.set_xlabel("distance to nearest face edge (px)")
        ax.set_ylabel("ratio"); ax.set_title(f"{name}: profile vs edge distance")
        print(name, "ratio std over (x,y):", float(mean_d.std()),
              " folded-64 std:", float(d.reshape(12, nside // 64, 64, nside // 64, 64).mean((0, 1, 3)).std()),
              " edge px ratio:", float(np.mean(prof[:2])), " centre:", float(np.mean(prof[-10:])))
    fig.savefig(os.path.join(a.out_dir, f"{var}_face_error.png"), dpi=100)
    np.save(os.path.join(a.out_dir, f"{var}_mse_nest.npy"), mse)


if __name__ == "__main__":
    main()
