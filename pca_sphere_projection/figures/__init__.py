"""Manuscript figure generation, organised per Figure / Panel.

Each figure module exposes a ``PANELS`` dict mapping a panel id (e.g. ``"A"``,
``"A_hesc"``) to a callable ``panel_fn(config: dict, out_root: pathlib.Path)
-> dict``.

The registry below collects all four figures into one place so the CLI in
``scripts/make_manuscript_figures.py`` can dispatch with ``--figure 3 --panel A_hesc``.

A panel function MUST:

- accept a ``config`` dict and an ``out_root`` Path;
- return a dict with keys ``figure`` (matplotlib Figure), ``data`` (dict of
  pandas DataFrames or numpy arrays, one CSV per entry), ``config`` (the
  resolved configuration written next to the PNG), and ``panel_id``
  (e.g. ``"Fig1A"``);
- be deterministic given ``config["seed"]``.

`common.save_panel(result, out_root)` then handles PNG + YAML + CSV writes.
"""

from . import common, fig1, fig2, fig3, fig4

REGISTRY = {
    1: fig1.PANELS,
    2: fig2.PANELS,
    3: fig3.PANELS,
    4: fig4.PANELS,
}

__all__ = ["common", "fig1", "fig2", "fig3", "fig4", "REGISTRY"]
