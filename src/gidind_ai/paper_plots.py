"""Manuscript-style vector plots generated from saved public benchmark results."""

from pathlib import Path
import csv
import numpy as np

COLORS = {"missing": "#1f77b4", "spurious": "#c44e52", "noise": "#1f4e79"}


def draw(ax, rows, panel, bands=False):
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    baseline = next(r for r in rows if r["arm"] == "peaks" and r["value"] == 0)
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#bdbdbd", linestyle="--", linewidth=0.65, alpha=0.8)
    ax.grid(which="minor", color="#d9d9d9", linestyle=":", linewidth=0.5, alpha=0.75)
    ax.tick_params(which="both", direction="out", width=0.7)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
    ax.set_ylabel("Percentage correctly predicted")

    def curve(rr, color):
        x = np.array([r["value"] for r in rr])
        y = np.array([r["fraction"] for r in rr])
        if bands:
            ax.fill_between(
                x,
                [r["ci_low"] for r in rr],
                [r["ci_high"] for r in rr],
                color=color,
                alpha=0.17,
                linewidth=0,
            )
        ax.plot(x, y, "o-", color=color, markersize=3.5, linewidth=1.5, markeredgewidth=0, zorder=4)

    if panel == "b":
        peak = sorted((r for r in rows if r["arm"] == "peaks"), key=lambda r: r["value"])
        extent = max(abs(r["value"]) for r in peak)
        ax.axvspan(-extent, 0, color=COLORS["missing"], alpha=0.08, linewidth=0)
        ax.axvspan(0, extent, color=COLORS["spurious"], alpha=0.08, linewidth=0)
        ax.axvline(0, color="#525252", linestyle="--", linewidth=0.9, zorder=3)
        curve([r for r in peak if r["value"] <= 0], COLORS["missing"])
        curve([r for r in peak if r["value"] >= 0], COLORS["spurious"])
        ax.set_xlim(-extent, extent)
        lower = min(r["ci_low"] if bands else r["fraction"] for r in peak)
        ax.set_ylim(min(0.6, np.floor(lower * 10) / 10), 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
        ax.set_xlabel("Peak perturbation count")
        for text, x, color in (
            ("Missing peaks", 0.25, COLORS["missing"]),
            ("Spurious peaks", 0.75, COLORS["spurious"]),
        ):
            ax.text(
                x,
                0.975,
                text,
                transform=ax.transAxes,
                color=color,
                ha="center",
                va="top",
                weight="bold",
                fontsize=9,
            )
        ax.legend(
            handles=[
                Line2D([0], [0], color=COLORS[k], lw=1.7, label=label)
                for k, label in (("missing", "Missing peaks"), ("spurious", "Spurious peaks"))
            ],
            loc="lower left",
            frameon=False,
            fontsize=8.5,
            handlelength=2.5,
        )
        zero = 0
    else:
        noise = sorted((r for r in rows if r["arm"] == "noise"), key=lambda r: r["value"])
        curve([dict(baseline, value=0)] + noise, COLORS["noise"])
        ax.set_xlim(0, max(r["value"] for r in noise))
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.005))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.0025))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "0" if abs(x) < 1e-10 else f"{x:.3f}")
        )
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.set_xlabel(r"Shift noise standard deviation, $\sigma_q$ ($\mathrm{\AA}^{-1}$)")
        zero = 0
    ax.plot(zero, baseline["fraction"], "o", color="black", markersize=4, zorder=6)
    ax.text(-0.18, 1.055, f"{panel})", transform=ax.transAxes, fontsize=14, weight="bold")
    ax.set_box_aspect(1)


def export(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next(
        (f for f in ("Calibri", "Carlito", "Arial", "DejaVu Sans") if f in available), "DejaVu Sans"
    )
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    for panels, bands in (("bc", False), ("b", False), ("c", False), ("bc", True)):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(8.2 if len(panels) == 2 else 4.2, 4.1),
            squeeze=False,
            layout="constrained",
        )
        for ax, panel in zip(axes.flat, panels):
            draw(ax, rows, panel, bands)
        name = f"figure3{panels}" + ("_95ci" if bands else "")
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(out / f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.07)
        plt.close(fig)
    return font


def figure3(directory):
    directory = Path(directory)
    with (directory / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in ["value", "fraction", "ci_low", "ci_high"]:
            row[name] = float(row[name])
    export(rows, directory)
