"""
Visualisation Utilities for the Fashion Search Engine.

Provides functions to display ranked search results as image grids and to
compare results from different queries side-by-side (useful for demonstrating
compositional understanding).
"""

import sys
import textwrap
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from config import RESULTS_DIR


def display_results(
    results: list[dict],
    query: str,
    save_path: Optional[Path] = None,
) -> None:
    """Display search results as a horizontal grid of images.

    Parameters
    ----------
    results : list[dict]
        Ranked candidate dicts.  Expected keys: ``image_path``, ``caption``,
        and at least one of ``final_score``, ``reranked_score``,
        ``fused_score``.  Optionally ``vlm_reasoning``.
    query : str
        The original user query (used as the figure super-title).
    save_path : Path, optional
        If provided the figure is saved to this path as a 300-dpi PNG.
        Otherwise ``plt.show()`` is called.
    """
    n = len(results)
    if n == 0:
        print("[Visualise] No results to display.")
        return

    n_cols = min(n, 5)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        f"Query: {query}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for idx, (ax, result) in enumerate(zip(axes, results[:n_cols])):
        image_path = result.get("image_path", "")

        # Determine best available score
        score = result.get(
            "final_score",
            result.get("reranked_score", result.get("fused_score", 0.0)),
        )

        # Attempt to load image
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
        except Exception:
            ax.text(
                0.5, 0.5, "Image\nnot found",
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes,
            )

        ax.set_title(f"#{idx + 1}  Score: {score:.3f}", fontsize=11, fontweight="bold")
        ax.axis("off")

        # Caption as x-label (wrapped)
        caption = result.get("caption", "")
        wrapped = "\n".join(textwrap.wrap(caption, width=35))
        ax.set_xlabel(wrapped, fontsize=8, labelpad=8)

        # VLM reasoning as annotation (if available)
        vlm_reasoning = result.get("vlm_reasoning")
        if vlm_reasoning:
            short_reasoning = textwrap.shorten(vlm_reasoning, width=60, placeholder="…")
            ax.annotate(
                short_reasoning,
                xy=(0.5, -0.02),
                xycoords="axes fraction",
                ha="center",
                fontsize=7,
                fontstyle="italic",
                color="grey",
            )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"[Visualise] Saved results to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def display_comparison(
    results1: list[dict],
    results2: list[dict],
    query1: str,
    query2: str,
    save_path: Optional[Path] = None,
) -> None:
    """Display two queries' results side-by-side (compositionality proof).

    Useful for demonstrating that "red shirt blue pants" and "blue shirt red
    pants" produce different rankings.

    Parameters
    ----------
    results1 : list[dict]
        Results for the first query.
    results2 : list[dict]
        Results for the second query.
    query1 : str
        First query string.
    query2 : str
        Second query string.
    save_path : Path, optional
        If provided, save the figure; otherwise display interactively.
    """
    n_cols = min(max(len(results1), len(results2)), 5)
    if n_cols == 0:
        print("[Visualise] No results to compare.")
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

    # Ensure axes is 2-D even for a single column
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        "Compositional Comparison",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    for row_idx, (results, query) in enumerate(
        [(results1, query1), (results2, query2)]
    ):
        axes[row_idx, 0].set_ylabel(
            f"Q: {textwrap.shorten(query, 40, placeholder='…')}",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            labelpad=80,
            va="center",
        )

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(results):
                result = results[col_idx]
                image_path = result.get("image_path", "")
                score = result.get(
                    "final_score",
                    result.get("reranked_score", result.get("fused_score", 0.0)),
                )
                try:
                    img = mpimg.imread(image_path)
                    ax.imshow(img)
                except Exception:
                    ax.text(
                        0.5, 0.5, "Image\nnot found",
                        ha="center", va="center", fontsize=12,
                        transform=ax.transAxes,
                    )

                ax.set_title(
                    f"#{col_idx + 1}  {score:.3f}",
                    fontsize=10,
                    fontweight="bold",
                )
                caption = result.get("caption", "")
                ax.set_xlabel(
                    "\n".join(textwrap.wrap(caption, width=30)),
                    fontsize=7,
                    labelpad=6,
                )
            else:
                ax.text(
                    0.5, 0.5, "—",
                    ha="center", va="center", fontsize=14,
                    transform=ax.transAxes,
                )

            ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"[Visualise] Saved comparison to {save_path}")
    else:
        plt.show()

    plt.close(fig)
