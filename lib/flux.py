import numpy as np
import numba as nb
import flow
import math

@nb.njit
def flux_impl(surface, neighbors, parse, flux, clamp = False):
    nr, nc = surface.shape
    surface = surface.flatten()

    for node in parse[-1::-1]:
        if clamp:
            flux[node] = max(flux[node], 0.0)

        total_slope = 0
        surface_node = surface[node]

        for nb in neighbors[node]:
            total_slope += (surface_node - surface[nb])

        if total_slope == 0:
            continue

        for nb in neighbors[node]:
            flux[nb] += flux[node] * (surface_node - surface[nb]) / total_slope

def flux(surface, source):
    neighbors = np.empty((surface.size,4), dtype = np.intp)
    parse = np.empty(surface.size, dtype = np.intp)
    flow.compute_neighbors_grad(surface, neighbors, parse)

    flux = source.copy().reshape(-1)
    flux_impl(surface, neighbors, parse, flux)

    return flux.reshape(surface.shape)
