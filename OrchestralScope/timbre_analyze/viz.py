from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt


def _apply_nature_style():
    # Compatible with different versions of seaborn style names; fallback to default if unavailable
    try:
        import matplotlib as mpl
        avail = set(getattr(mpl.style, "available", []))
        if "seaborn-v0_8-whitegrid" in avail:
            plt.style.use("seaborn-v0_8-whitegrid")
        elif "seaborn-whitegrid" in avail:
            plt.style.use("seaborn-whitegrid")
        else:
            plt.style.use("default")
    except Exception:
        plt.style.use("default")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.edgecolor": "#333333",
        "grid.color": "#e0e0e0",
    })


def plot_embedding_scatter(
    out_dir: Path,
    name: str,
    X: np.ndarray,
    colors: Optional[List[str]] = None,
    point_labels: Optional[List[str]] = None,
    class_letters: Optional[List[str]] = None,
    dominant_class_indices: Optional[List[int]] = None,
    letter_to_color: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    annotate_3d: bool = False,
    legend_loc: str = "best",
):
    out_dir = Path(out_dir) / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    _apply_nature_style()
    fig = plt.figure(figsize=(7.2, 5.2))
    if X.shape[1] == 2:
        ax = fig.add_subplot(111)
        sc = ax.scatter(
            X[:, 0],
            X[:, 1],
            c=colors if colors is not None else "#4C78A8",
            s=36,
            alpha=0.95,
            linewidths=0.6,
            edgecolors="white",
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
    else:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=colors if colors is not None else "#4C78A8",
            s=26, alpha=0.95, linewidths=0.4, edgecolors="white",
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        # Optional: annotate labels in 3D as well (may be crowded)
        if annotate_3d and point_labels is not None:
            for (x, y, z), lbl in zip(X[:, :3], point_labels):
                ax.text(x, y, z, lbl, fontsize=6, color="#333333")

    if title:
        ax.set_title(title)

    # Add text labels (avoid overcrowding, slight offset)
    if point_labels is not None and X.shape[1] == 2:
        jitter = (np.random.RandomState(42).randn(len(point_labels), 2)) * 0.01
        for (x, y), jt, lbl in zip(X[:, :2], jitter, point_labels):
            ax.text(
                x + jt[0], y + jt[1], lbl,
                fontsize=8, color="#333333",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    # Generate legend (by dominant class letters)
    if class_letters is not None and letter_to_color is not None:
        handles = []
        import matplotlib.patches as mpatches
        for L in class_letters:
            col = letter_to_color.get(L, "#999999")
            handles.append(mpatches.Patch(color=col, label=L))
        ax.legend(handles=handles, title="Dominant timbre", loc=legend_loc, frameon=True)

    fig.tight_layout()
    # Save both unlabeled and labeled versions
    fig.savefig(out_dir / f"{name}.png")
    if point_labels is not None and (X.shape[1] == 2 or annotate_3d):
        fig.savefig(out_dir / f"{name}_labeled.png")
    plt.close(fig)

    # Additionally export point label mapping for typesetting: embedding/{name}_labels.csv
    try:
        import pandas as pd
        rows = []
        for idx in range(X.shape[0]):
            rows.append({
                "index": idx,
                "label": point_labels[idx] if point_labels is not None else "",
                "x": float(X[idx, 0]),
                "y": float(X[idx, 1]),
                **({"z": float(X[idx, 2])} if X.shape[1] >= 3 else {}),
                "color": colors[idx] if colors is not None else "",
            })
        df = pd.DataFrame(rows)
        (Path(out_dir).parent / "embedding").mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(out_dir).parent / "embedding" / f"{name}_labels.csv", index=False)
    except Exception:
        pass


def plot_complexity_timeline(out_dir: Path, track_name: str, beat_times: np.ndarray, C_raw: np.ndarray, C_smooth: np.ndarray, peak: Optional[Dict] = None):
    out_dir = Path(out_dir) / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    
    # Calculate percentage time axis
    total_duration = beat_times[-1] if len(beat_times) > 0 else 1.0
    time_percentages = (beat_times / total_duration) * 100
    
    ax.plot(time_percentages[:len(C_raw)], C_raw, color="#90CAF9", linewidth=1.0, label="C_raw")
    ax.plot(time_percentages[:len(C_smooth)], C_smooth, color="#1E88E5", linewidth=1.5, label="C_smooth")
    
    if peak is not None:
        # Calculate percentage position of peak
        peak_percentage = (peak['peak_time'] / total_duration) * 100
        ax.axvline(peak_percentage, color="#E53935", linestyle="--", linewidth=1.0)
        ax.scatter([peak_percentage], [peak['peak_C']], color="#E53935", zorder=3)
        ax.text(peak_percentage, peak['peak_C'], f" peak={peak['peak_C']:.2f}", color="#E53935", fontsize=9, va="bottom", ha="left")
    
    ax.set_xlabel("Time (%)")
    ax.set_ylabel("Local complexity (per beat window)")
    ax.set_title(f"Complexity timeline: {track_name}")
    ax.legend(loc="upper right")
    
    # Set x-axis ticks as percentages
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))  # 0%, 20%, 40%, 60%, 80%, 100%
    
    fig.tight_layout()
    fig.savefig(out_dir / f"{track_name}_complexity.png")
    plt.close(fig)



def plot_hierarchical_from_distance(out_dir: Path, track_names: List[str], D: np.ndarray, method: str = "average", title: Optional[str] = None) -> None:
    """
    Perform hierarchical clustering based on precomputed distance matrix and plot dendrogram.
    Parameters:
      - out_dir: Output directory root
      - track_names: Row/column labels (track names)
      - D: NxN distance matrix (symmetric, main diagonal is 0)
      - method: Linkage method (default 'average')
    Output:
      - figs/hclust_combined.png: Dendrogram
      - distance/combined_hclust_steps.csv: Linkage steps matrix
    """
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except Exception:
        return

    try:
        d_condensed = squareform(D, checks=False)
    except Exception:
        d_condensed = D

    Z = linkage(d_condensed, method=method)

    try:
        import pandas as pd
        (Path(out_dir) / "distance").mkdir(parents=True, exist_ok=True)
        df_steps = pd.DataFrame(Z, columns=["cluster1", "cluster2", "distance", "count"])
        df_steps.to_csv(Path(out_dir) / "distance" / "combined_hclust_steps.csv", index=False)
    except Exception:
        pass

    out_figs = Path(out_dir) / "figs"
    out_figs.mkdir(parents=True, exist_ok=True)
    _apply_nature_style()
    fig = plt.figure(figsize=(9.6, 6.4))
    ax = fig.add_subplot(111)
    # Keep only the first word of track names (separated by space or underscore)
    try:
        import re
        labels_use = []
        for n in track_names:
            tok = re.split(r"[ _]+", str(n).strip())
            first = tok[0] if tok and tok[0] else str(n)
            labels_use.append(first)
    except Exception:
        labels_use = track_names
    dendrogram(
        Z,
        labels=labels_use,
        orientation="top",
        distance_sort="descending",
        show_leaf_counts=True,
        ax=ax,
    )
    ax.set_title(title or "Hierarchical clustering (combined distance, average linkage)")
    ax.set_xlabel("Tracks")
    ax.set_ylabel("Distance")
    # Adjust x-axis label font size
    ax.tick_params(axis='x', labelsize=5)
    fig.tight_layout()
    fig.savefig(out_figs / "hclust_combined.png")
    plt.close(fig)

