import cv2
import sys
import json
import numba
import pickle
import skimage
import skimage.io
import numpy as np
from scipy import ndimage
from scipy.ndimage import morphology as morpho
from scipy import interpolate
import time
from QTree import QTreeNode
from matplotlib.patches import Ellipse
import matplotlib.colors as mcols

benchmark_before = time.time()
def benchStart(title):
    global benchmark_before
    benchmark_before = time.time()
    print(title + "...", file=sys.stderr)

benchmark_after = 0
def benchEnd():
    global benchmark_after
    benchmark_after = time.time()
    print("\t%.3fs" % (benchmark_after - benchmark_before), file=sys.stderr)

def closestValid(loc, obs):
    """
    Returns closest valid location (no obstacle in obs) from input loc
    """
    candidates = [loc]
    added = [loc[0] + loc[1] * obs.shape[0]]
    while len(candidates) > 0:
        coords = candidates.pop(0)
        if coords[0] < 0 or coords[1] < 0 or coords[0] >= len(obs) or coords[1] >= len(obs):
            continue
        if obs[int(coords[1]), int(coords[0])] == 0:
            return np.array(coords)
        for nbs in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nxt_coord = (coords[0] + nbs[0], coords[1] + nbs[1])
            nxt_idx = nxt_coord[0] + nxt_coord[1] * obs.shape[0]
            if nxt_idx not in added:
                candidates.append(nxt_coord)
                added.append(nxt_idx)
    return False

def computePath(qtree, start_p, end_p, scale):
    """
    Compute path between start and end in a scaled quad tree
    """
    start_coords = closestValid(start_p // scale, qtree.im)
    end_coords = closestValid(end_p // scale, qtree.im)

    start = qtree.findLeaf(*start_coords)
    end = qtree.findLeaf(*end_coords)

    qtree.reset()
    if start.aStar(end):
        segs = []
        curr = end
        while curr != start:
            if len(segs) == 0:
                hsz = curr.sz >> 1
                segs.append(((curr.x + hsz) * scale, (curr.y + hsz) * scale))
            hsz = curr.previous.sz >> 1
            segs.append(((curr.previous.x + hsz) * scale, (curr.previous.y + hsz) * scale))
            curr = curr.previous
        return [start_p] + segs[1:-1][::-1] + [end_p]
    return None

def bspline(cv, n=100, degree=3, periodic=False):
    """ From: https://stackoverflow.com/a/35007804
    Calculate n samples on a bspline

    cv :      Array ov control vertices
    n  :      Number of samples to return
    degree:   Curve degree
    periodic: True - Curve is closed
    False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)
    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)
    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1)
    else:
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)
    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)
    # Calculate result
    return np.array(interpolate.splev(u, (kv,cv.T,degree))).T

def makeBackground(terrain, water = None, colnames = ["white", "black"], scale = 1):
    cmapcols = np.array([mcols.to_rgb(x) for x in colnames])
    cmapxs = np.linspace(0, 1, len(cmapcols))
    if scale != 1:
        height, width = terrain.shape
        terrain = cv2.resize(terrain, (height // scale, width // scale))
        if water is not None:
            water = cv2.resize(water, (height // scale, width // scale))
    background = np.dstack(([np.interp(terrain / terrain.max(), cmapxs, cmapcols[:,i]) for i in range(np.size(cmapcols, 1))]))
    if water is not None:
        background[water == 1] = mcols.to_rgb("royalblue")
    return background

def computePercentileBounds(density, percentile_min, percentile_max = -1):
    """
    Compute values of percentiles in a density map
    """
    if percentile_max == -1:
        percentile_max = 1.0 - percentile_min
    values = np.sort(density.flatten())
    vmin = values[int((len(values)-1) * percentile_min)]
    vmax = values[int((len(values)-1) * percentile_max)]
    return vmin, vmax

def pltEllipses(ax, ells, edge, fill="none"):
    """
    Plot a collection of ellipses in pyplot axes
    """
    for ell in ells:
        artist = Ellipse(ell[0:2], ell[2], ell[3], np.rad2deg(ell[4]))
        ax.add_artist(artist)
        artist.set_edgecolor(edge)
        artist.set_facecolor(fill)

def cv2Circle(canvas, loc, sz, col, thickness=-1):
    """
    Wrapper to draw a circle on an image with opencv
    """
    cv2.circle(canvas, tuple(loc.astype(int) // scale), int(np.ceil(sz / 2)), col, thickness)

def cv2Rectan(canvas, loc, sz, col, thickness=-1):
    """
    Wrapper to draw a centered rectangle on an image with opencv
    """
    loc = loc.astype(int) // scale
    hsz = int(np.ceil(sz/2))
    cv2.rectangle(canvas, tuple(loc - hsz), tuple(loc + hsz), col, thickness)

@numba.njit()
def pt2idx(im, pt):
    """
    Encode coordinates in a single integer
    """
    return im.shape[1] * pt[0] + pt[1]
@numba.njit()
def idx2pt(im, idx):
    """
    Decode coordinates from a single integer
    """
    return (idx // im.shape[1], idx % im.shape[1])
