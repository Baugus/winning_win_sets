"""
figure_generator.py

Generate 2D charts of voters and status-quo with circles and shaded regions
where at least a given fraction of voter-circles overlap.

Dependencies: numpy, matplotlib
Install: pip install numpy matplotlib

Usage: modify parameters in the example at the bottom or import functions
from this module in other scripts.
"""

import math
from pathlib import Path
from typing import Sequence, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


Point = Tuple[float, float]


def draw_win_set(voters: Sequence[Point],
                    status_quo: Point = (0.0, 0.0),
                    threshold: float = 0.5,
                    grid_res: int = 700,
                    margin: float = 0.3,
                    figsize: Tuple[float, float] = (7, 7),
                    circle_edgecolor: str = "#27476E",
                    circle_facecolor: str = "none",
                    circle_linewidth: float = 1.,
                    overlap_color: str = "#bdbdbd",
                    overlap_alpha: float = 0.7,
                    overlap_order: int = 1,
                    voter_marker: str = "o",
                    voter_color: str = "#2f6dad",
                    voter_size: float = 20,
                    alienation_ring_radius: Optional[float] = None,
                    alienation_ring_linewidth: float = 0.8,
                    alienation_threshold_criterion: str = "winset",
                    hide_alienated_circles: bool = False,                    
                    draw_alienation_rings: bool = True,                    
                    alienated_voter_color: str = "#e78f3c",
                    status_marker: str = "o",
                    status_color: str = "k",
                    status_size: float = 20,
                    labels: Optional[Sequence[str]] = None,
                    status_label: Optional[str] = None,
                    label_offset: Optional[float] = None,
                    draw_axes: bool = True,
                    axis_origin: Optional[Point] = None,
                    axis_color: str = "#999999",
                    axis_linewidth: float = 0.6,
                    axis_linestyle: str = "-",
                    axis_label_left: Optional[str] = "L",
                    axis_label_right: Optional[str] = "R",
                    axis_label_bottom: Optional[str] = "Auth",
                    axis_label_top: Optional[str] = "Dem",
                    show_winset_centroid: bool = False,
                    winset_centroid_label: Optional[str] = None,
                    winset_centroid_color: str = "#572700",
                    winset_centroid_marker: str = "D",
                    winset_centroid_size: float = 20,
                    savepath: Optional[str] = None,
                    additional_save_format: Optional[str] = None,
                    transparent_background: bool = True,
                    show: bool = True,
                    ref_points: Sequence[Point] = None,
                    ref_points_labels: Optional[str] = None,
                    ref_point_marker: str = "D",
                    ref_point_size: float = 20,
                    ref_point_color: str = "#572700", 
                    fig_pad: float = None,
                    x_limit: float = None,
                    y_limit: float = None
                    ):
    """
    Generate a 2D spatial voting figure with voter circles, win set regions, and optional indicators.

    Core Parameters:
    - voters: sequence of (x,y) points representing voter ideal points
    - status_quo: (x,y) point for the status quo (default: origin)
    - threshold: fraction (0-1) of voters whose circles must overlap to shade the win set (default: 0.5)
    - grid_res: resolution of sampling grid for win set computation; higher = more precise (default: 700)
    - margin: padding around data as fraction of data range (default: 0.3)
    - figsize: tuple (width, height) for figure size in inches (default: (7, 7))

    Voter Circle Parameters (distance-to-status-quo circles):
    - circle_edgecolor: color of voter circles (default: "#27476E" dark blue)
    - circle_facecolor: fill color (default: "none")
    - circle_linewidth: stroke width of voter circles (default: 1.0)

    Win Set Display:
    - overlap_color: color of shaded win set region (default: "#bdbdbd" grey)
    - overlap_alpha: transparency of win set shading (default: 0.7)

    Voter Point Parameters:
    - voter_marker: matplotlib marker style for voter points (default: "o" circle)
    - voter_color: color of non-alienated voter points (default: "#2f6dad" blue)
    - voter_size: size of voter point markers (default: 20)

    Alienation Ring Parameters:
    - alienation_ring_radius: optional radius for alienation rings around voters; None = off (default: None)
    - alienation_ring_linewidth: stroke width of alienation rings (default: 0.8)
    - draw_alienation_rings: whether to visually draw the rings (default: True)
    - hide_alienated_circles: if True, hide voter circles and recolor points for alienated voters (default: False)
    - alienation_threshold_criterion: determines alienation detection method (default: "winset")
        * "winset": voter is alienated if ring doesn't overlap the shaded win set region
        * "status_quo": voter is alienated if ring doesn't contain the status quo point
        * "centroid": voter is alienated if ring doesn't contain the win set centroid
    - alienated_voter_color: color for alienated voter points when hide_alienated_circles=True (default: "#e78f3c" orange)

    Status Quo Parameters:
    - status_marker: matplotlib marker style for status quo point (default: "o" circle)
    - status_color: color of status quo point (default: "k" black)
    - status_size: size of status quo marker (default: 20)
    - status_label: optional label for status quo point; None = no label (default: None)

    Label Parameters:
    - labels: optional sequence of labels for voter points (default: None)

    Output Parameters:
    - savepath: optional output path to save the figure (default: None)
    - additional_save_format: optional second lossless format (e.g., "png", "svg", "tiff")
      saved with the same base filename as savepath (default: None)
    - transparent_background: whether saved figures use a transparent background (default: True)
    - label_offset: offset distance for labels from points; auto-computed if None (default: None)

    Axis Parameters:
    - draw_axes: whether to draw x/y axes through origin (default: True)
    - axis_origin: point through which axes pass; None = use status quo (default: None)
    - axis_color: color of axis lines (default: "#999999" grey)
    - axis_linewidth: stroke width of axis lines (default: 0.6)
    - axis_linestyle: line style ("−", "−−", ":", etc.) (default: "−")
    - axis_label_left: label for left axis end; None = no label (default: "L")
    - axis_label_right: label for right axis end; None = no label (default: "R")
    - axis_label_bottom: label for bottom axis end; None = no label (default: "Auth")
    - axis_label_top: label for top axis end; None = no label (default: "Dem")

    Win Set Centroid Parameters:
    - show_winset_centroid: whether to plot the centroid point of the win set region (default: False)
    - winset_centroid_label: optional label for the centroid point; None = no label (default: None)
    - winset_centroid_color: color of centroid marker (default: "#572700" brown)
    - winset_centroid_marker: matplotlib marker style for centroid (default: "D" diamond)
    - winset_centroid_size: size of centroid marker (default: 20)

    Output Parameters:
    - savepath: path to save figure (PDF/SVG/PNG); None = don't save (default: None)
    - show: whether to call plt.show() to display figure (default: True)

    Returns:
    - tuple (fig, ax): matplotlib Figure and Axes objects
    """

    voters = np.asarray(voters, dtype=float)
    sq = np.asarray(status_quo, dtype=float)
    n = len(voters)
    if n == 0:
        raise ValueError("At least one voter is required")

    # compute radii
    radii = np.linalg.norm(voters - sq, axis=1)

    # bounding box (include circles)
    xs = np.concatenate([voters[:, 0], [sq[0]]])
    ys = np.concatenate([voters[:, 1], [sq[1]]])
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    maxr = float(radii.max())
    dx = xmax - xmin
    dy = ymax - ymin
    if fig_pad is None:
        # implement dynamic padding
        pad = max(maxr, max(dx, dy) * margin)
    elif type(fig_pad) == int or type(fig_pad) == float:
        pad = fig_pad
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    # implement optional cropping
    if x_limit is not None:
        if abs(xmin) > x_limit:
            xmin = -x_limit
        if abs(xmax) > x_limit:
            xmax = x_limit

    if y_limit is not None:
        if abs(ymin) > y_limit:
            ymin = -y_limit
        if abs(ymax) > y_limit:
            ymax = y_limit


    # sample grid
    X = np.linspace(xmin, xmax, grid_res)
    Y = np.linspace(ymin, ymax, grid_res)
    XX, YY = np.meshgrid(X, Y)

    # coverage count
    counts = np.zeros_like(XX, dtype=int)
    # print(counts)
    for (vx, vy), r in zip(voters, radii):
        # print(vx)
        # print(vy)
        mask = (XX - vx) ** 2 + (YY - vy) ** 2 <= (r + 1e-12) ** 2
        # print(mask)
        counts += mask.astype(int)

    # threshold (number of voters required)
    required = math.ceil(threshold * n)
    cover_mask = counts >= required

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    if transparent_background:
      fig.patch.set_alpha(0)
      ax.set_facecolor("none")

    # filled overlap region using contourf (vector-friendly)
    fraction = counts.astype(float) / max(1.0, n)
    levels = [threshold, 1.0]
    cf = ax.contourf(XX, YY, fraction, levels=levels, colors=[overlap_color], alpha=overlap_alpha, zorder=overlap_order)

    # optional axes through a chosen origin (defaults to status quo)
    if draw_axes:
        origin = np.asarray(axis_origin if axis_origin is not None else status_quo, dtype=float)
        ox, oy = float(origin[0]), float(origin[1])
        ax.plot([xmin, xmax], [oy, oy], color=axis_color, linewidth=axis_linewidth, linestyle=axis_linestyle, zorder=0)
        ax.plot([ox, ox], [ymin, ymax], color=axis_color, linewidth=axis_linewidth, linestyle=axis_linestyle, zorder=0)
        # axis end labels (offset outward to clear the endpoints)
        label_offset_x = (xmax - xmin) * 0.02
        label_offset_y = (ymax - ymin) * 0.02
        if axis_label_left is not None:
            ax.text(xmin - label_offset_x, oy, axis_label_left, ha="right", va="center", color=axis_color)
        if axis_label_right is not None:
            ax.text(xmax + label_offset_x, oy, axis_label_right, ha="left", va="center", color=axis_color)
        if axis_label_bottom is not None:
            ax.text(ox, ymin - label_offset_y, axis_label_bottom, ha="center", va="top", color=axis_color)
        if axis_label_top is not None:
            ax.text(ox, ymax + label_offset_y, axis_label_top, ha="center", va="bottom", color=axis_color)

    # identify alienated voters (if enabled and radius is provided)
    alienated_voters_set = set()
    if hide_alienated_circles and alienation_ring_radius is not None:
        for i, (vx, vy) in enumerate(voters):
            is_alienated = False
            
            if alienation_threshold_criterion == "status_quo":
                # check if alienation ring contains status quo
                dist_to_sq = np.linalg.norm(np.array([vx, vy]) - sq)
                is_alienated = dist_to_sq > alienation_ring_radius
            
            elif alienation_threshold_criterion == "centroid":
                # check if alienation ring contains win set centroid
                if show_winset_centroid:
                    winset_points = np.column_stack(np.where(cover_mask))
                    if len(winset_points) > 0:
                        yi_mean, xi_mean = winset_points.mean(axis=0)
                        xi_scaled = X[int(xi_mean)] if int(xi_mean) < len(X) else X[-1]
                        yi_scaled = Y[int(yi_mean)] if int(yi_mean) < len(Y) else Y[-1]
                        centroid = np.array([xi_scaled, yi_scaled])
                        dist_to_centroid = np.linalg.norm(np.array([vx, vy]) - centroid)
                        is_alienated = dist_to_centroid > alienation_ring_radius
            
            elif alienation_threshold_criterion == "winset":
                # check if alienation ring contains any part of the winset
                winset_points = np.column_stack(np.where(cover_mask))
                if len(winset_points) > 0:
                    # map grid indices to coordinates
                    winset_coords = np.column_stack([X[winset_points[:, 1].astype(int)],
                                                      Y[winset_points[:, 0].astype(int)]])
                    # compute distance from voter to each winset point
                    voter_pos = np.array([vx, vy])
                    dists_to_winset = np.linalg.norm(winset_coords - voter_pos, axis=1)
                    # alienated if no winset points are within alienation ring
                    is_alienated = np.all(dists_to_winset > alienation_ring_radius)
                else:
                    is_alienated = True  # no winset, so alienated
            
            if is_alienated:
                alienated_voters_set.add(i)

    # draw circles and points
    for i, ((vx, vy), r) in enumerate(zip(voters, radii)):
        # skip drawing circle for alienated voters if flag is set
        if i not in alienated_voters_set:
            circ = Circle((vx, vy), r, facecolor=circle_facecolor, edgecolor=circle_edgecolor, linewidth=circle_linewidth)
            ax.add_patch(circ)

    # optional alienation rings around voters
    if alienation_ring_radius is not None and draw_alienation_rings:
        for (vx, vy) in voters:
            alienation = Circle((vx, vy), alienation_ring_radius, facecolor="none", edgecolor=voter_color, linewidth=alienation_ring_linewidth, linestyle="dashed", zorder=5)
            ax.add_patch(alienation)

    # plot voters (separate alienated and non-alienated if applicable)
    if hide_alienated_circles and alienated_voters_set:
        # plot non-alienated voters
        non_alienated_indices = [i for i in range(len(voters)) if i not in alienated_voters_set]
        if non_alienated_indices:
            non_alienated_voters = voters[non_alienated_indices]
            ax.scatter(non_alienated_voters[:, 0], non_alienated_voters[:, 1], marker=voter_marker, color=voter_color, s=voter_size, zorder=10)
        # plot alienated voters in different color
        alienated_indices = list(alienated_voters_set)
        alienated_voters_points = voters[alienated_indices]
        ax.scatter(alienated_voters_points[:, 0], alienated_voters_points[:, 1], marker=voter_marker, color=alienated_voter_color, s=voter_size, zorder=10)
    else:
        # plot all voters in the same color
        ax.scatter(voters[:, 0], voters[:, 1], marker=voter_marker, color=voter_color, s=voter_size, zorder=10)
    
    # plot optional reference points
    if ref_points is not None:
        ref_points = np.asarray(ref_points, dtype=float)
        ax.scatter(ref_points[:, 0], ref_points[:, 1], marker=ref_point_marker, color=ref_point_color, s=ref_point_size, zorder=10)

    # Plot status quo
    ax.scatter([sq[0]], [sq[1]], marker=status_marker, color=status_color, s=status_size, zorder=11)

    # optional labels (with computed offset if not provided)
    if label_offset is None:
        label_offset = max(dx, dy) * 0.03

    if labels is not None:
        for (x, y), lab in zip(voters, labels):
            ax.text(x + label_offset, y + label_offset, lab, fontsize=10, fontstyle="italic", color=voter_color, verticalalignment="center")

    if status_label is not None:
        ax.text(sq[0] + label_offset, sq[1] + label_offset, status_label, fontsize=10, fontstyle="italic", color=status_color, verticalalignment="center")

    if ref_points_labels is not None and ref_points is not None:
        for (x, y), lab in zip(ref_points, ref_points_labels):
            ax.text(x + label_offset, y + label_offset, lab, fontsize=10, fontstyle="italic", color=ref_point_color, verticalalignment="center")

    # optional win set centroid
    if show_winset_centroid:
        # compute the centroid of points in the win set region
        winset_points = np.column_stack(np.where(cover_mask))
        if len(winset_points) > 0:
            # map grid indices back to coordinate space
            yi_mean, xi_mean = winset_points.mean(axis=0)
            xi_scaled = X[int(xi_mean)] if int(xi_mean) < len(X) else X[-1]
            yi_scaled = Y[int(yi_mean)] if int(yi_mean) < len(Y) else Y[-1]
            centroid = np.array([xi_scaled, yi_scaled])
            ax.scatter([centroid[0]], [centroid[1]], marker=winset_centroid_marker, color=winset_centroid_color, s=winset_centroid_size, zorder=12)
            if winset_centroid_label is not None:
                ax.text(centroid[0] + label_offset, centroid[1] + label_offset, winset_centroid_label, fontsize=10, fontstyle="italic", color=winset_centroid_color, verticalalignment="center")

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300, transparent=transparent_background)
        if additional_save_format:
            format_name = additional_save_format.lower().lstrip(".")
            allowed_lossless_formats = {"pdf", "png", "svg", "tif", "tiff"}
            if format_name not in allowed_lossless_formats:
                raise ValueError(
                    f"additional_save_format must be one of {sorted(allowed_lossless_formats)}, got '{additional_save_format}'"
                )

            save_path_obj = Path(savepath)
            alternate_savepath = save_path_obj.with_suffix(f".{format_name}")
            if alternate_savepath != save_path_obj:
                fig.savefig(alternate_savepath, bbox_inches="tight", dpi=300, transparent=transparent_background)

    # if show:
    #     plt.show()

    return fig, ax

# FIXME: Feature requests
# - a live editor where you can click and drag points around and have it save the parameters
# - different classes of voters
# - win sets calculated separately for either
    # - different classes of voters, or
    # - different sets of voters based on parameters (i.e. only non-alienated voters)
# - calculate the size of the win set
# - better measure of alienation that allows for alienation if the min of some set of objs is farther away than x
# - auto calc various voting results
# - n-dimensions
# - label the axes to show the values along each dimension corresponding to voter's ideal points
# - allow individual voters to have different parameters, including:
    # - assumed status quo,
    # - costs of voting
    # - reservation utility


if __name__ == "__main__":
  save_dir = Path(r"C:\Users\Baugus\OneDrive - Duke University\Projects\chilean_constitutional_process\08_presentations\PCS26\figures")


  # Figure 1: Example Win Set
  status_quo = (0, 0)
  status_quo_label = '$P_{sq}$'
  centroid_label = r'$\bar{p}$'
  # centroid_label = '$\mathbb{E}[u(p)]$'

  voters = [
              (-2.1, 1.25), 
              (1.3, 0.44), 
              (-0.35, -0.85),
              # (-0.3, -0.9),
              ]
              

  labels = [
              "A",
              'B', 
              'C',
              ]

  save_path = save_dir / Path('example_winset_2D.pdf')

  fig, ax = draw_win_set(voters, 
                              status_quo=status_quo, 
                              labels=labels, 
                              savepath=save_path,
                              status_label=status_quo_label,
                              circle_linewidth = 1,
                              axis_origin=(0,0),
                              )
  plt.close(fig)

  print('saved fig 1')