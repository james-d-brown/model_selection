"""Writing Plotly figures to standalone HTML files.

The figure container is given the figure's own pixel size explicitly;
Plotly's responsive sizing proved unreliable across export paths.
"""
from __future__ import annotations

import os

import plotly.graph_objects as go


def write_standalone_html(fig: go.Figure, path: str, page_title: str) -> None:
    """Write one figure to a standalone HTML page at a fixed pixel size."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    width = fig.layout.width or 1400
    height = fig.layout.height or 800
    fragment = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": False})
    with open(path, "w") as f:
        f.write(
            f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{page_title}</title></head>"
            f"<body style='margin:0; padding:20px;'>"
            f"<div style='width:{width}px; height:{height}px;'>{fragment}</div>"
            f"</body></html>"
        )


def write_multi_figure_html(figs: dict[str, go.Figure], path: str, page_title: str) -> None:
    """Write several figures, each under its own heading, to one HTML page."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{page_title}</title></head>"
                f"<body style='margin:0; padding:20px;'>\n")
        for name, fig in figs.items():
            width, height = fig.layout.width or 1400, fig.layout.height or 800
            fragment = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": False})
            f.write(f"<h2>{name}</h2>\n")
            f.write(f"<div style='width:{width}px; height:{height}px;'>{fragment}</div>\n")
        f.write("</body></html>\n")
