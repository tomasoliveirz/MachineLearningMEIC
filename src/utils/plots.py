#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared plotting utilities for consistent visualization across the project.

Includes directory management, string formatting for labels, and style setup.
"""

from pathlib import Path
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist.
    
    Args:
        path: Path to directory to create
        
    Returns:
        The same Path object (for chaining)
        
    Example:
        fig_dir = ensure_dir(Path("reports/figures"))
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sentence_case(s: str) -> str:
    """Convert string to sentence case for plot labels.
    
    First character uppercase, rest lowercase. Used to make plot labels
    consistently formatted.
    
    Args:
        s: Input string
        
    Returns:
        String in sentence case
        
    Example:
        >>> sentence_case("ROOKIE DEVELOPMENT")
        'Rookie development'
        >>> sentence_case("regular-season win%")
        'Regular-season win%'
    """
    if not s:
        return s
    s = str(s)
    return s[0].upper() + s[1:].lower()


def save_fig(fig, path: Path, dpi: int = 160, **kwargs) -> None:
    """Save figure with consistent settings.
    
    Args:
        fig: Matplotlib figure object
        path: Output path
        dpi: Resolution (default: 160)
        **kwargs: Additional arguments for fig.savefig()
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    plt.close(fig)


def setup_default_style(style: str = "whitegrid") -> None:
    """Setup consistent matplotlib/seaborn style.
    
    Args:
        style: Seaborn style name (default: "whitegrid")
    """
    try:
        import seaborn as sns
        sns.set(style=style)
    except ImportError:
        # If seaborn not available, use basic matplotlib style
        plt.style.use("seaborn-v0_8-whitegrid" if style == "whitegrid" else "default")


def setup_plot_rcparams(dpi: int = 140, title_size: int = 14, 
                        label_size: int = 12, tick_size: int = 10) -> None:
    """Setup consistent matplotlib rcParams for plots.
    
    Args:
        dpi: Figure DPI
        title_size: Axis title font size
        label_size: Axis label font size
        tick_size: Tick label font size
    """
    plt.rcParams.update({
        "figure.dpi": dpi,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
    })

