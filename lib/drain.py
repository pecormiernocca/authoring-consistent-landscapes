import numpy as np
import numba
import flux
import flow
import math

def compute(surface, source, fill=True):
    if fill:
        surface = fill_holes(surface)
    return flux.flux(surface, source)

@numba.njit
def water_height_impl(surface, neighbors, parse, area, water, boundary):
    nr, nc = surface.shape
    surface = surface.reshape(-1)
    water = water.reshape(-1)
    area = area.reshape(-1)
    boundary = boundary.reshape(-1)

    for node in parse:
        if boundary[node]:
            water[node] = 0
            continue

        slope_proxy = 2*surface[node] - surface[neighbors[node, 0]] - surface[neighbors[node, 1]] - water[neighbors[node, 0]] - water[neighbors[node, 1]]

        #solve h (2h + slope_proxy) = A
        delta = slope_proxy*slope_proxy + 8 * area[node]
        water[node] = .25 * (-slope_proxy + math.sqrt(delta))

def water_height(surface, area, boundary = None):
    if boundary is None:
        boundary = np.ones(surface.shape, dtype = np.bool)
        boundary[1:-1, 1:-1] = False
    water = np.empty_like(surface)
    neighbors = np.empty((surface.size,2), dtype = np.intp)
    parse = np.empty(surface.size, dtype = np.intp)
    flow.compute_neighbors_grad(surface, neighbors, parse)
    water_height_impl(surface, neighbors, parse, area, water, boundary)
    return water
