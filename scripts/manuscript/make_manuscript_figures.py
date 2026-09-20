"""Manuscript figure orchestrator (thin dispatcher).

Each panel lives in `pca_sphere_projection/figures/figN.py` and exports a
`PANELS` dict mapping panel id to a function `panel_fn(config, out_root)`
that returns `{panel_id, figure_n, figure, data, config}`.

Run all figures:

    python scripts/manuscript/make_manuscript_figures.py

Run one figure:

    python scripts/manuscript/make_manuscript_figures.py --figure 3

Run one panel:

    python scripts/manuscript/make_manuscript_figures.py --figure 3 --panel A_hesc
    python scripts/manuscript/make_manuscript_figures.py --figure 1 --panel A

Override hyperparameters from the CLI:

    python scripts/manuscript/make_manuscript_figures.py --figure 1 --panel E \
        --override perturbation_degrees=10 n_perturbations=300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pca_sphere_projection import figures  # noqa: E402
from pca_sphere_projection.figures import common  # noqa: E402


def _parse_overrides(items):
    """Parse `key=value` overrides; values are auto-cast to int/float/bool/str."""
    out = {}
    if not items:
        return out
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--override entries must be key=value (got {it!r})")
        k, v = it.split("=", 1)
        for caster in (int, float):
            try:
                out[k] = caster(v); break
            except ValueError:
                pass
        else:
            if v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            else:
                out[k] = v
    return out


def _run_panel(figure_n: int, panel_id: str, override: dict,
                out_root: Path, manifest: list):
    panel_fn = figures.REGISTRY[figure_n][panel_id]
    t0 = time.time()
    try:
        result = panel_fn(config=override, out_root=out_root)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[Fig{figure_n} {panel_id}] FAILED: {exc}")
        # Write a 'data not present' card so the slot is occupied.
        result = common.data_not_present_result(
            f"Fig{figure_n}{panel_id}" if not panel_id.startswith(("A_","B_","C_","D_","summary"))
            else f"Fig{figure_n}{panel_id}",
            figure_n, f"Fig {figure_n} {panel_id} — error", tb[:400])
    common.save_panel(result, out_root, registry=manifest)
    dt = time.time() - t0
    print(f"  {result['panel_id']:24s} -> {out_root.name}/Fig{figure_n}/{result['panel_id']}.png  ({dt:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=int, choices=[1, 2, 3, 4, 0], default=0,
                        help="0 (default) = all figures")
    parser.add_argument("--panel", type=str, default=None,
                        help="run only this panel id (within --figure)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "figures")
    parser.add_argument("--override", nargs="*", default=None,
                        help="key=value overrides for the panel config")
    args = parser.parse_args()

    common.apply_publication_style()
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    override = _parse_overrides(args.override)
    manifest = []

    figs = [args.figure] if args.figure else [1, 2, 3, 4]
    for fn in figs:
        panels = figures.REGISTRY[fn]
        if args.panel:
            if args.panel not in panels:
                raise SystemExit(
                    f"unknown panel {args.panel!r} for Fig{fn}; "
                    f"valid: {sorted(panels)}")
            ids = [args.panel]
        else:
            ids = list(panels)
        print(f"\n=== Fig {fn} ({len(ids)} panel(s)) ===")
        for pid in ids:
            _run_panel(fn, pid, override, out_root, manifest)

    manifest_path = out_root / "manifest.txt"
    manifest_path.write_text("\n".join(sorted(manifest)) + "\n")
    print(f"\nManifest -> {manifest_path}")
    print(f"{len(manifest)} PNG(s) generated under {out_root}.")


if __name__ == "__main__":
    main()
