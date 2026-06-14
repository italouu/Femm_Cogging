import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_grid(Field):
    # --- plot ---
    plt.figure(figsize=(8,6))
    plt.imshow(Field, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(f"Mag")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_ang_grid(Field, r_in, r_ext, ang_1 = 0, ang_2 = 120):

    n_r, n_a = Field.shape  # infere pelas dimensões
    ang_1 = np.deg2rad(ang_1)
    ang_2 = np.deg2rad(ang_2)

    r_edges = np.linspace(r_in, r_ext, n_r+1)
    a_edges = np.linspace(ang_1, ang_2, n_a+1)
    R, A = np.meshgrid(r_edges, a_edges, indexing="ij")

    X = R * np.cos(A)
    Y = R * np.sin(A)

    fig, ax = plt.subplots(subplot_kw={"aspect":"equal"})
    pcm = ax.pcolormesh(X, Y, Field, cmap="viridis", shading="auto")

    cbar = fig.colorbar(pcm, ax=ax)
    ax.set_title("Magnitude")
    plt.show()

def curl(field):

    Fx = field[0, :, :]
    Fy = field[1, :, :]

    # derivadas periódicas
    dFy_dx = Fy.roll(-1, dims=0) - Fy.roll(1, dims=0)
    dFx_dy = Fx.roll(-1, dims=1) - Fx.roll(1, dims=1)

    # rotacional z
    curl_z = dFy_dx - dFx_dy

    return curl_z

def plot_quadtree(depth, data=None, dim=(1, 1), color_by="depth",
                  child_order="NW_NE_SW_SE", lw=0.8, show_values=False, ax=None,
                  cmap=None, norm=None):
    """
    Plot quadtree leaf rectangles from a (depth-per-leaf) vector (and optional data),
    WITHOUT reconstructing a full-resolution grid.

    Assumptions:
      - Base domain is split into `dim = (H, W)` coarse cells.
      - Leaves are listed in a DFS order. When a cell is refined, children are visited in:
          child_order="NW_NE_SW_SE" (default) or "SW_SE_NW_NE".
      - `depth[i]` is the depth of leaf i.

    Params:
      depth: 1D list/np.array of ints (depth per leaf)
      data:  optional 1D list/np.array same length as depth (value per leaf)
      dim:   (H, W) number of coarse cells
      color_by: "depth" or "data"
      child_order: "NW_NE_SW_SE" or "SW_SE_NW_NE"
      lw: line width
      show_values: write depth/data value at cell center
      ax: matplotlib axis (optional)

    Returns:
      ax
    """

    depth = np.asarray(depth, dtype=int)
    if data is not None:
        data = np.asarray(data)
        if len(data) != len(depth):
            raise ValueError("data and depth must have the same length.")

    H, W = dim

    # --- build list of leaf rectangles by decoding depth stream ---
    # Each stack item: (x, y, w, h, current_depth)
    stack = []
    for r in reversed(range(H)):
        for c in reversed(range(W)):
            stack.append((float(c), float(r), 1.0, 1.0, 0))

    rects = []  # (x, y, w, h, d, leaf_idx)
    i = 0

    def push_children(x, y, w, h, d):
        w2, h2 = w / 2.0, h / 2.0
        nd = d + 1

        # origin top-left, y increases downward (we'll invert axis)
        NW = (x,      y,      w2, h2, nd)
        NE = (x+w2,   y,      w2, h2, nd)
        SW = (x,      y+h2,   w2, h2, nd)
        SE = (x+w2,   y+h2,   w2, h2, nd)

        if child_order == "NW_NE_SW_SE":
            order = [NW, NE, SW, SE]
        elif child_order == "SW_SE_NW_NE":
            order = [SW, SE, NW, NE]
        else:
            raise ValueError("Unknown child_order")

        for ch in reversed(order):  # LIFO -> process first child first
            stack.append(ch)

    while stack:
        x, y, w, h, dcur = stack.pop()
        if i >= len(depth):
            raise ValueError("Depth vector ended early (not enough leaves).")

        dleaf = int(depth[i])
        if dleaf == dcur:
            rects.append((x, y, w, h, dleaf, i))
            i += 1
        elif dleaf > dcur:
            push_children(x, y, w, h, dcur)
        else:
            raise ValueError(f"Inconsistent depth sequence: got {dleaf} but cell is at depth {dcur}.")

    if i != len(depth):
        raise ValueError("Depth vector has extra entries (too many leaves).")

    # --- setup axis ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # --- colormap ---
    if color_by == "data" and data is not None:
        _norm   = norm  if norm  is not None else colors.Normalize(vmin=float(np.min(data)), vmax=float(np.max(data)))
        _cmap   = cmap  if cmap  is not None else "viridis"
        mapper  = cm.ScalarMappable(norm=_norm, cmap=_cmap)
        get_val = lambda d, idx: float(data[idx])
    else:
        dvals   = depth.astype(float)
        _norm   = norm  if norm  is not None else colors.Normalize(vmin=float(dvals.min()), vmax=float(dvals.max()))
        _cmap   = cmap  if cmap  is not None else "Greys"
        mapper  = cm.ScalarMappable(norm=_norm, cmap=_cmap)
        get_val = lambda d, idx: float(d)

    # --- draw ---
    for x, y, w, h, dleaf, idx in rects:
        val = get_val(dleaf, idx)
        ax.add_patch(Rectangle((x, y), w, h, linewidth=lw,
                               edgecolor="none", facecolor=mapper.to_rgba(val)))
        if show_values:
            ax.text(x + w/2, y + h/2, str(int(val) if color_by == "depth" else val),
                    ha="center", va="center", fontsize=8)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    return ax
