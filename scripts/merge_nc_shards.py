"""Merge the per-rank NetCDF shards (<dir>/0.nc, 1.nc, ...) into one file sorted by time.

Usage: python scripts/merge_nc_shards.py out/exp1_step180000_test merged.nc
"""

import argparse
import glob
import os

import netCDF4 as nc
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir")
    p.add_argument("output")
    args = p.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.input_dir, "[0-9]*.nc")),
        key=lambda f: int(os.path.basename(f)[:-3]),
    )
    srcs = [nc.Dataset(f) for f in paths]
    times = np.concatenate([s["time"][:] for s in srcs])
    # (shard, index within shard) for every time step, in global time order
    where = [(i, j) for i, s in enumerate(srcs) for j in range(len(s["time"]))]
    order = np.argsort(times, kind="stable")
    assert len(np.unique(times)) == len(times), "duplicate time steps across shards"

    ref = srcs[0]
    with nc.Dataset(args.output, "w", format=ref.data_model) as out:
        out.setncatts({k: ref.getncattr(k) for k in ref.ncattrs()})
        for name, dim in ref.dimensions.items():
            out.createDimension(name, None if name == "time" else len(dim))
        for name, v in ref.variables.items():
            o = out.createVariable(name, v.dtype, v.dimensions)
            o.setncatts({k: v.getncattr(k) for k in v.ncattrs()})
            if "time" not in v.dimensions:
                o[:] = v[:]
        out["time"][:] = times[order]
        data_vars = [n for n, v in ref.variables.items() if "time" in v.dimensions and n != "time"]
        for name in data_vars:
            for k, g in enumerate(order):
                i, j = where[g]
                out[name][k] = srcs[i][name][j]
            print(f"wrote {name}")
    for s in srcs:
        s.close()
    print(f"{len(times)} time steps -> {args.output}")


if __name__ == "__main__":
    main()
