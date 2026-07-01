"""
Subprocess entry point for OpenMP-isolated model fits.

Run as: python -m src.models._isolated_worker <in.pkl> <out.pkl>

The in-pickle holds (task, args). We dispatch on task and write the result to
out.pkl. This module is launched in a fresh interpreter (see
experiment._run_isolated) so each booster gets a clean OpenMP runtime — the
reliable macOS fix for the libomp/libgomp duplicate-runtime abort.

Thread-limiting env vars are set by the parent BEFORE this process starts, so
they take effect before numpy/sklearn/booster import here.
"""
from __future__ import annotations

import pickle
import sys


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "rb") as f:
        task, args = pickle.load(f)

    if task == "oos":
        from src.models.experiment import _oos_worker
        result = _oos_worker(*args)
    elif task == "cv":
        from src.models.experiment import _cv_worker
        result = _cv_worker(*args)
    elif task == "hpo_fold":
        from src.models.hpo import _hpo_outer_fold_worker
        result = _hpo_outer_fold_worker(*args)
    else:
        raise ValueError(f"Unknown isolated task: {task}")

    with open(out_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
