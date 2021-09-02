## helpers to order nodes dependancies by height ""

import numba as nb
import numpy as np

@nb.njit
def compute_neighbors(surface, neighbors, neighbors_size, parse, nbs):
    """
    In: surface (nx*ny): elevation data
        nbs: coordinates of neighbors to parse, as tupe of tuples
    Out neighbors[node, :neighbors_size[node, 0]] :                    lower nodes
        neighbors[neighbors_size[node, 0], :neighbors_size[node, 1]] : upper nodes
        parse: parse order
    """

    nr, nc = surface.shape
    surface = surface.reshape(-1)

    stack = np.empty_like(surface, dtype = np.intp)
    stack_size = 0

    for node in range(surface.size):
        rcv_end = 0
        don_end = 0

        r = node // nc
        c = node %  nc

        surface_node = surface[node]

        for n_ir, n_ic in nbs:
            n_ir += r
            n_ic += c

            if n_ir < 0 or n_ic < 0 or n_ir >= nr or n_ic >= nc:
                continue

            nb = n_ir * nc + n_ic
            nb_surface = surface[nb]

            # donnor
            if nb_surface > surface_node:
                neighbors[node, don_end] = nb
                don_end += 1

            # receiver
            if nb_surface < surface_node:
                neighbors[node, don_end] = neighbors[node, rcv_end]
                neighbors[node, rcv_end] = nb
                rcv_end += 1
                don_end += 1

        neighbors_size[node, 0] = rcv_end
        neighbors_size[node, 1] = don_end

        if not rcv_end: #rcv_end = 0 ; add node as local minimum
            stack[stack_size] = node
            stack_size += 1

    visibility = np.zeros(surface.size, dtype = np.intp)

    parse_size = 0

    while stack_size:
        stack_size -= 1
        node = stack[stack_size]

        parse[parse_size] = node
        parse_size+=1

        for nb in neighbors[node, neighbors_size[node, 0]: neighbors_size[node, 1]]:
            visibility[nb] += 1
            if visibility[nb] == neighbors_size[nb, 0]:
                stack[stack_size] = nb
                stack_size += 1

@nb.njit
def inverse(parse, neighbors, neighbors_size):
    cparse = parse[-1::-1].copy()

    for n in range(parse.size):
        parse[n] = cparse[n]

        for i in range(neighbors_size[n, 1]//2):
            ii = neighbors_size[n, 1] - 1 - i
            t = neighbors[n,i]
            neighbors[n,i] = neighbors[n,ii]
            neighbors[n,ii] = t

        neighbors_size[n, 0] = neighbors_size[n, 1] - neighbors_size[n, 0]

def compute_neighbors_d4(surface, neighbors, neighbors_size, parse):
    return compute_neighbors(surface, neighbors, neighbors_size, parse,
                            ((-1, 0), (0, -1), (0, 1), (1, 0)))

def compute_neighbors_d8(surface, neighbors, neighbors_size, parse):
    return compute_neighbors(surface, neighbors, neighbors_size, parse,
                            ((-1, -1), (-1, 0), (-1, 1),
                             ( 0, -1),          ( 0, 1),
                             ( 1, -1), ( 1, 0), ( 1, 1)))

@nb.njit
def compute_neighbors_grad(surface, neighbors, parse):
    """
    In: surface (nx*ny): elevation data
    Out:
    neighbors[node, 0] : lower neighbor within same row or node
    neighbors[node, 1] : second lower neighbor within same row or node
    neighbors[node, 2] : lower neighbor  within same col or node
    neighbors[node, 3] : second lower neighbor  within same col or node

    nb 0, 1 : in direction of d/dx =(d/dc)

    parse: parse order
    """
    nr, nc = surface.shape
    surface = surface.reshape(-1)

    stack = np.empty_like(surface, dtype = np.intp)
    stack_size = 0

    donnors = np.zeros((stack.size, 5), dtype = np.intp) # donnors,num  donnors
    n_rcv =  np.zeros_like(stack)

    for node in range(surface.size):
        rcv_end = 0
        don_end = 0

        r = node // nc
        c = node %  nc

        neighbors[node, 0] = node
        neighbors[node, 1] = node
        neighbors[node, 2] = node
        neighbors[node, 3] = node

        icm = max(0, c-1)+nc*r
        icp = min(nc-1, c+1)+nc*r
        irm = max(0, r-1)*nc +c
        irp = min(nr-1, r+1)*nc+c

        iminc, imaxc = (icm, icp) if surface[icm] < surface[icp] else (icp, icm)
        iminr, imaxr = (irm, irp) if surface[irm] < surface[irp] else (irp, irm)

        s = surface[node]

        if surface[iminc] < s:
            neighbors[node, 0] = iminc
        if surface[imaxc] < s:
            neighbors[node, 1] = imaxc

        if surface[iminr] < s:
            neighbors[node, 2] = iminr
        if surface[imaxr] < s:
            neighbors[node, 3] = imaxr

        #local min, add to stack for latter parse
        if neighbors[node, 0] == node and neighbors[node, 2] == node:
            stack[stack_size] = node
            stack_size += 1

        for nb in neighbors[node]:
            if nb != node:
                n_rcv[node] += 1
                donnors[nb, donnors[nb,4]] = node
                donnors[nb,4] += 1

    visibility = np.zeros(surface.size, dtype = np.intp)

    parse_size = 0

    while stack_size:
        stack_size -= 1
        node = stack[stack_size]

        parse[parse_size] = node
        parse_size+=1

        for nb in donnors[node, : donnors[node, 4]]:
            visibility[nb] += 1
            if visibility[nb] == n_rcv[nb]:
                stack[stack_size] = nb
                stack_size += 1
