"""Plot style and saving figures.

    figs = FigureSaver("1_flux_ul", "2_data")
    fig, ax = plt.subplots()
    ...
    figs.save(fig, "counts_map")   # outputs/figures/1_flux_ul/2_data/counts_map.{png,pdf}
"""
from pathlib import Path

import matplotlib as mpl

from . import paths

# Okabe-Ito, colour-blind safe; same palette as simulst.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def use_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (6.4, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.prop_cycle": mpl.cycler(color=COLORS),
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "image.cmap": "viridis",
    })


class FigureSaver:
    """Saves figures of one notebook under outputs/figures/<study>/<notebook>/."""

    def __init__(self, study: str, notebook: str, formats=("png", "pdf"), root: Path | None = None):
        self.folder = Path(root or paths.FIGURES) / study / notebook
        self.formats = tuple(formats)
        use_style()

    def save(self, fig, name: str) -> list[Path]:
        self.folder.mkdir(parents=True, exist_ok=True)
        out = [self.folder / f"{name}.{ext}" for ext in self.formats]
        for p in out:
            fig.savefig(p)
        return out
