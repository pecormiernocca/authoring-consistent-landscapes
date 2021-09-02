import matplotlib.colors as mcols
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import skimage
import skimage.morphology
import numba
import cv2
import sys
import pathlib

sys.path.append("./lib")
import QTree
import Animal
import utils
from Region import Region

sols_tautavel = ["J", "L"]

sol = "J"
resource_path = "resources/"
water_threshold = -2
density_threshold = 0.20
tree_obstacle_threshold = 0.3
terrain_cell_width = 1

map_exits = [[0, 1650], # Exits
            [0, 2660],
            [0, 3600],
            [3060, 0],
            [4096, 420],
            [1957, 1121], # Fords
            [1973, 1104],
            [1201, 2578],
            [1231, 2562],
            [1356, 3106],
            [1383, 3093]]
shrub_idx = [0, 2]
tree_idx = [1, 3, 8, 10, 11, 12, 14, 15]
plant_idx = [4, 5, 6, 7, 9, 13, 16, 17]
if sol == "J":
    resource_path += "tautavel_J/"
    estimated_pop = {"Bison": 2,
                    "Elk": 71,
                    "Deer": 59,
                    "Reindeer": 4,
                    "Mouflon": 10,
                    "Rhino": 4,
                    "Horse": 3}
    estimated_carn = {"Bear": 2,
                    "Wolf": 3,
                    "Lion": 1,
                    "Lynx": 2,
                    "Fox": 2}
elif sol == "L":
    resource_path += "tautavel_L/"
    estimated_pop = {"Bison": 1,
                    "Elk": 12,
                    "Deer": 2,
                    "Reindeer": 78,
                    "Mouflon": 8,
                    "Rhino": 1,
                    "Horse": 1}
    estimated_carn = {"Bear": 2,
                    "Wolf": 3,
                    "Lion": 1,
                    "Lynx": 1,
                    "Fox": 1}
pathlib.Path(resource_path + "visu").mkdir(parents=True, exist_ok=True)

###############################################################################
# Load all resources, including heightmap, water, plant densities, etc.
###############################################################################
utils.benchStart("Loading images")
smooth = cv2.imread(resource_path + "../raw/dem_filled.tif", cv2.IMREAD_UNCHANGED)
drain = cv2.imread(resource_path + "../raw/water.tif", cv2.IMREAD_UNCHANGED)
drain_log = np.log(drain)
idx = (drain_log > water_threshold)
water = np.zeros(drain.shape)
water[idx] = 1

bg_scale = 4
background = utils.makeBackground(smooth, water, ['darkgreen', 'gold', 'darkred', 'black'], bg_scale)
background_bw = utils.makeBackground(smooth, water, ['white', 'black'], bg_scale)

plant_density_count = max(max(shrub_idx), max(tree_idx), max(plant_idx))+1
species_herb = Animal.species_herb
species_carn = Animal.species_carn

plants = []
for i in range(plant_density_count):
    plant = skimage.io.imread(resource_path + "flora/plant-%d.tif" % i).astype(np.float32)
    plants.append(plant)
utils.benchEnd()

###############################################################################
# Extract chains of water pixels along rivers
# Similar to vectorizing the water bodies
###############################################################################
utils.benchStart("Extracting water chains")

def decimateChains(chains, max_err = 200):
    """
    Decimate chains with Douglas-Peucker algorithm
    max_err is the maximum distance allowed from a point to the simplification
    """
    newchains = []
    for chain in chains:
        vs = chain - chain[0]
        angles = np.arctan2(vs[:,1], vs[:,0])
        vas = angles - angles[-1]
        ds = np.linalg.norm(vs, axis=1)
        errs = np.abs(np.sin(vas) * ds)
        id_far = np.argmax(errs)
        if errs[id_far] > max_err:
            newchains += decimateChains([chain[:id_far+1], chain[id_far:]], max_err)
        else:
            newchains.append(chain)
    return newchains

@numba.njit
def numbaFillChain(visited, init, pt):
    chain = [init]
    sx, sy = pt
    while True:
        chain.append([sx, sy])
        if neighbs[sy, sx] != 2:
            return np.array(chain)
        visited[sy, sx] = True
        for kidx in range(9):
            dx = (kidx % 3) - 1
            dy = int(kidx / 3) - 1
            if dx == 0 and dy == 0:
                continue
            nsx = sx + dx
            nsy = sy + dy
            if nsx < 0 or nsx >= szw or nsy < 0 or nsy >= szh:
                continue
            if not ma[nsy, nsx]: # Not on the medial axis
                continue
            if visited[nsy, nsx]: # Already registered
                continue
            sx = nsx
            sy = nsy
            break
    return np.array(chain)

def fillChain(junctions):
    visited = np.zeros(ma.shape, np.bool8)
    chains = []
    # Loop through each junction (has neighbors but not 2)
    for it in range(len(junctions)):
        sx, sy = junctions[it]
        visited[sy, sx] = True
        # Find neighbors
        for dx, dy in kernel_idx:
            if sx+dx < 0 or sx+dx >= szw or sy+dy < 0 or sy+dy >= szh:
                continue
            if not ma[sy+dy, sx+dx]: # Not on the medial axis
                continue
            if visited[sy+dy, sx+dx]: # Already registered
                continue
            chain = numbaFillChain(visited, [sx, sy], [sx+dx, sy+dy])
            if len(chain) > 2:
                chains.append(chain)
    return chains

szh, szw = water.shape
ma, dfield = skimage.morphology.medial_axis(water, return_distance=True)

# 8-connectivity kernel and relative indexes
kernel = np.ones((3,3), dtype=int)
kernel[1,1] = 0
kernel_idx = np.vstack(np.where(kernel)[::-1]).T-1

neighbs = np.zeros(ma.shape, dtype=int)
for x, y in kernel_idx:
    neighbs[0+(y<0):szh-(y>0), 0+(x<0):szw-(x>0)] += ma[0+(y>0):szh-(y<0), 0+(x>0):szw-(x<0)]
neighbs *= ma
junctions = np.vstack(np.where((neighbs > 0) & (neighbs != 2))[::-1]).T
chains = fillChain(junctions)
chains = decimateChains(chains, 50)
utils.benchEnd()

###############################################################################
# Compute summarized density maps for vegetation
###############################################################################
utils.benchStart("Computing presence maps")
shrubs_presence = np.max([plants[i] for i in shrub_idx], axis=0) > density_threshold
trees_presence = np.max([plants[i] for i in tree_idx], axis=0) > density_threshold
plants_presence = np.max([plants[i] for i in plant_idx], axis=0) > density_threshold
trees_obstacles = np.max([plants[i] for i in tree_idx], axis=0) > tree_obstacle_threshold
presence_maps = [shrubs_presence, plants_presence, trees_presence]
presence_colors = ["red", "green", "purple"]
presence_names = ["Shrubs", "Plants", "Trees"]
presence_idx = [shrub_idx, plant_idx, tree_idx]

# Export merged density maps
cmap = plt.get_cmap("Greens")
percentile = 0.05
for i, idxs in enumerate([plant_idx, shrub_idx, tree_idx]):
    plt.subplot(1, 3, i+1)
    density = np.max([plants[i] for i in idxs], axis=0)
    vmin, vmax = utils.computePercentileBounds(density, percentile)
    norm = mcols.Normalize(vmin, vmax)
    out = cmap(norm(density))[:,:,:3]
    out[water == 1] = mcols.to_rgb("royalblue")
    plt.imshow(out)
    plt.title(presence_names[i])
    plt.axis("off")
plt.tight_layout()
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
plt.savefig(resource_path + "visu/plant_densities.png")

im = smooth / terrain_cell_width
us = np.zeros(im.shape)
vs = np.zeros(im.shape)
us[:,:-1] = im[:,1:] - im[:,:-1]
vs[:-1,:] = im[1:,:] - im[:-1,:]
ds = np.sqrt(us**2 + vs**2)

im2 = np.ones(im.shape)
im2[ds >= .5] = 0
im2[water != 0] = 0
im2[trees_obstacles != 0] = 0
utils.benchEnd()

###############################################################################
# Compute accessibility maps
###############################################################################
utils.benchStart("Extracting regions")
@numba.njit
def numbaApplyAlongAxis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@numba.njit
def numbaMean(array, axis):
    return numbaApplyAlongAxis(np.mean, axis, array)

@numba.njit
def numbaExtractRegions(lbls, nb_lbls, region_size_threshold, accessibility):
    region_pxs = [np.zeros(1, np.int64)]
    region_ellipses = [[0.0]]
    access_regions = [0]
    min_val = 0
    big_lbls = np.ones(lbls.shape) * -1

    vals = lbls.flatten()
    nbins = nb_lbls+2
    order = np.argsort(vals)
    sep = np.sort(np.searchsorted(vals[order], np.arange(nb_lbls+2)))
    for j in range(1, nb_lbls+1):
        if sep[j+1] - sep[j] <= region_size_threshold:
            continue
        reg_ids = order[sep[j]:sep[j+1]]
        region_x = reg_ids % lbls.shape[0]
        region_y = (reg_ids / lbls.shape[0]).astype(np.int32)

        if accessibility is not None:
            region_pxs.append(reg_ids)
            region_xy = np.vstack((region_x, region_y)).T
            centers = numbaMean(region_xy, 0).astype(np.int32)
            local_xy = region_xy - centers
            cov = np.cov(local_xy, rowvar=False)
            eigvals, eigvecs = np.linalg.eig(cov)
            confidence = np.sqrt(eigvals * 4)
            ellipse = [centers[0], centers[1],
                        2*confidence[0], 2*confidence[1],
                        np.arctan2(eigvecs[1,0], eigvecs[0,0])]
            region_ellipses.append(ellipse)

            res = accessibility[region_y[0], region_x[0]]
            access_regions.append(res)

        for idxi in range(len(region_y)):
            big_lbls[region_y[idxi], region_x[idxi]] = min_val
        min_val += 1
    return big_lbls, region_ellipses, access_regions, region_pxs

def extractRegions(image, region_size_threshold=10000, accessibility=None):
    """
    Extract contiguous regions with a flood-fill algorithm
    """
    lbls, nb_lbls = skimage.measure.label(image, return_num=True, connectivity = 1)
    bl, re, ar, pxs = numbaExtractRegions(lbls, nb_lbls, region_size_threshold, accessibility)
    return bl, np.array(re[1:]), np.array(ar[1:]), pxs[1:]

accessibility, garb1, garb2, accessibility_px = extractRegions(im2)

nb_regions = int(accessibility.max() + 1)
vege_lbls = []
vege_ellipses = []
vege_regions = []
vege_pxs = []
reg_sz_thresh = 10000 / (terrain_cell_width ** 2)
for presence_map in presence_maps:
    presence_map = presence_map.astype(np.int)
    presence_map[(ds >= .5) | (water != 0) | (trees_obstacles != 0)] = 0

    lbls, region_ellipses, access_regions, region_px = extractRegions(presence_map, region_size_threshold=reg_sz_thresh, accessibility=accessibility)
    lbls[lbls == -1] = -np.inf
    lbls[water != 0] = -1
    vege_lbls.append(lbls)
    vege_ellipses.append(region_ellipses)
    vege_regions.append(access_regions)
    vege_pxs.append(region_px)
utils.benchEnd()

###############################################################################
# Simplify water chains and convert them to nodes
###############################################################################
utils.benchStart("Mapping water nodes")

def chainsToEllipses(chains, dfield):
    s0 = np.zeros((len(chains), 2))
    s1 = np.zeros((len(chains), 2))
    ellipses = np.zeros((len(chains), 5))
    for i, chain in enumerate(chains):
        v = np.array(chain[-1]) - chain[0]
        xy = (v * 0.5) + chain[0]
        ellipses[i, :2] = xy
        ellipses[i, 2] = np.linalg.norm(v)
        ellipses[i, 3] = 5 * 2 * np.mean(dfield[chain[:,1], chain[:,0]])
        ellipses[i, 4] = np.arctan2(*v[::-1])
        s0[i] = chain[0]
        s1[i] = chain[-1]
    return ellipses, (s0, s1)

def closestSegments(segments, pts):
    """
    Returns the index of the closest segments to the given points
    """
    mat = np.zeros((len(pts), len(segments[0])))
    s0, s1 = segments
    u = s1 - s0
    l2 = np.sum(u ** 2, 1)
    for i in range(len(s0)):
        raw = np.dot(pts - s0[i], u[i]) / l2[i]
        weight = (raw < 0) * (-raw) + (raw > 1) * (raw - 1)
        ts = np.clip(raw, 0, 1)
        projs = s0[i] + np.atleast_2d(ts).T * u[i]
        mat[:,i] = np.sum((projs - pts) ** 2, 1) + weight
    return np.argmin(mat, 1)

# Create a map of pixels on land and adjacent to water
drinks = np.zeros(accessibility.shape, dtype=int)
for i in range(9):
    x, y = (i % 3)-1, int(i / 3)-1
    if x == 0 and y == 0:
        continue
    drinks[0+(y<0):szh-(y>0), 0+(x<0):szw-(x>0)] += (water[0+(y>0):szh-(y<0), 0+(x>0):szw-(x<0)] != 0)
drinks *= (accessibility >= 0)

# Compute closest ellipses for each of the pixels near water
contact_where = np.where(drinks!=0)
contact = np.vstack(contact_where[::-1]).T
water_ellipses, water_segments = chainsToEllipses(chains, dfield)
dto = closestSegments(water_segments, contact)

# Store ellipses within in each region
region_vals = accessibility[contact_where]
nb_lbls = int(np.max(region_vals))
dtos = [None] * (nb_lbls+1)
nbins = nb_lbls+2
order = np.argsort(region_vals)
sep = np.sort(np.searchsorted(region_vals[order], np.arange(nb_lbls+2)))
for j in range(0, nb_lbls+1):
    reg_ids = order[sep[j]:sep[j+1]]
    dtos[j] = np.unique(dto[reg_ids]).astype(np.int)
utils.benchEnd()

###############################################################################
# Store nodes per region
###############################################################################
utils.benchStart("Organizing data per region")
regions = []
for reg_i in range(nb_regions):
    veges = []
    veges_px = []
    for vege_i in range(len(presence_maps)):
        idx = (vege_regions[vege_i] == reg_i)
        sub_ells = vege_ellipses[vege_i][idx]
        veges.append(sub_ells)
        sub_pxs = [vege_pxs[vege_i][i] for i,take in enumerate(idx) if take]
        veges_px.append(sub_pxs)
    sub_water = np.zeros((0, 5))
    water_px = []
    if reg_i < len(dtos):
        sub_water = water_ellipses[dtos[reg_i]]
        for ellipse_i in dtos[reg_i]:
            water_px.append(contact[(dto == ellipse_i) & (region_vals == reg_i)])
    regions.append(Region(reg_i, veges, veges_px, sub_water, water_px))
utils.benchEnd()

total_pop = sum(estimated_pop.values()) + sum(estimated_carn.values())
estimated_density = {name:num/sum(estimated_pop.values()) for name,num in estimated_pop.items()}
carn_density = {name:num/total_pop for name,num in estimated_carn.items()}

###############################################################################
# Compute density maps for animals
###############################################################################
utils.benchStart("Competition algorithm for animals")
for region in regions:
    region.resetMaxResources()
total = 0
instance_count = np.zeros(len(species_herb))
tgt_density = [estimated_density[animal.name] for animal in species_herb]
region_pop = np.zeros((len(species_herb), len(regions)))
region_herds = np.empty((len(species_herb), len(regions)), dtype=list)
for y in range(len(species_herb)):
    for x in range(len(regions)):
        region_herds[y, x] = []

while True:
    idx, grp_sz = Animal.getCandidateId(instance_count, total, tgt_density, species_herb)
    animal = species_herb[idx]
    fit = Animal.fitness(regions, animal, grp_sz, cell_width_sq=terrain_cell_width**2)
    region_idx = np.argmax(fit)
    if int(fit[region_idx]) == 0:
        break
    region = regions[region_idx]
    region.resources[0] -= animal.shrub_req * (Animal.YEARLY_RATIO / (terrain_cell_width ** 2)) * grp_sz
    region.resources[1] -= animal.plant_req * (Animal.YEARLY_RATIO / (terrain_cell_width ** 2)) * grp_sz
    region.resources[2] -= animal.tree_req * (Animal.YEARLY_RATIO / (terrain_cell_width ** 2)) * grp_sz
    instance_count[idx] += grp_sz
    total += grp_sz
    region_pop[idx, region_idx] += grp_sz
    region_herds[idx, region_idx].append(grp_sz)

tgt_carn_density = [carn_density[animal.name] for animal in species_carn]
region_surplus = (region_pop.T * [sp.surplus * np.mean(sp.mass) for sp in species_herb]).T / 365.0
carn_herds = np.empty((len(species_carn), len(regions)), dtype=list)
carn_pop = np.zeros((len(species_carn), len(regions)))
carn_count = np.zeros(len(species_carn))
for y in range(len(species_carn)):
    for x in range(len(regions)):
        carn_herds[y, x] = []
while True:
    idx, grp_sz = Animal.getCandidateId(carn_count, total, tgt_carn_density, species_carn)
    animal = species_carn[idx]
    carn_fitness = Animal.fitness(regions, species_carn[idx], grp_sz, region_surplus, cell_width_sq=terrain_cell_width**2)
    best_reg = np.argmax(carn_fitness)
    if int(carn_fitness[best_reg]) == 0:
        break
    region_surplus[:,best_reg] -= animal.herbivores_req * grp_sz * region_surplus[:,best_reg] / region_surplus[:,best_reg].sum()
    carn_count[idx] += grp_sz
    total += grp_sz
    carn_pop[idx, best_reg] += grp_sz
    carn_herds[idx, best_reg].append(grp_sz)
utils.benchEnd()

utils.benchStart("Exporting animal density maps")
meat_map = np.zeros(accessibility.shape)
density_cmap = plt.get_cmap("Oranges")
resized_maps = []
sub_species = np.array([3, 1, 8])
for specie_i in np.arange(len(species_herb)):
    specie = species_herb[specie_i]
    density_map = np.zeros(accessibility.shape)
    for region_i, reg in enumerate(regions):
        pop = region_pop[specie_i, region_i]
        for vege_i, veges_px in enumerate(reg.veges_px):
            type_ratio = specie.reqs[vege_i] / specie.sum_reqs
            if type_ratio == 0:
                continue
            szs = np.array([len(x) for x in veges_px])
            ratios = type_ratio * (szs / np.sum(szs))
            weights = pop * ratios
            for idx, weight in enumerate(weights):
                pxs = veges_px[idx]
                ys = pxs // np.size(accessibility, 0)
                xs = pxs % np.size(accessibility, 0)
                density_map[ys, xs] += weight / szs[idx]
                meat_map[ys, xs] += np.mean(species_herb[0].mass) * weight / szs[idx]
    density_map = cv2.resize(density_map, background.shape[:2], interpolation=cv2.INTER_NEAREST)
    resized_maps.append(density_map)
max_density = np.max([dmap.max() for dmap in resized_maps])
for i, density_map in enumerate(resized_maps):
    if i not in sub_species:
        continue
    specie = species_herb[i]
    out = background_bw.copy()
    if density_map.max() > 0:
        colored_density = density_cmap(density_map / max_density)
        idxs = density_map > 0
        out[idxs] = colored_density[idxs][:,:3]
    plt.subplot(1, len(sub_species), np.where(sub_species == i)[0][0] + 1)
    plt.title(specie.name)
    plt.imshow(out)
    plt.axis("off")

for specie_i in sub_species[sub_species > len(species_herb)]:
    carn_i = specie_i - len(species_herb)
    specie = species_carn[carn_i]
    density_map = np.zeros(accessibility.shape)
    for region_i, reg in enumerate(regions):
        pop = carn_pop[carn_i, region_i]
        if pop == 0:
            continue
        for vege_i, veges_px in enumerate(reg.veges_px):
            type_ratio = specie.reqs[vege_i] / specie.sum_reqs
            if type_ratio == 0:
                continue
            szs = np.array([len(x) for x in veges_px])
            ratios = type_ratio * (szs / np.sum(szs))
            weights = pop * ratios
            for idx, weight in enumerate(weights):
                pxs = veges_px[idx]
                ys = pxs // np.size(accessibility, 0)
                xs = pxs % np.size(accessibility, 0)
                density_map[ys, xs] += weight / szs[idx]
        type_ratio = specie.reqs[-1] / specie.sum_reqs
        access_idxs = accessibility == region_i
        density_map[access_idxs] += pop * meat_map[access_idxs] * type_ratio / np.sum(meat_map[access_idxs])
    density_map = cv2.resize(density_map, background.shape[:2], interpolation=cv2.INTER_NEAREST)
    colored_density = density_cmap(density_map / max_density)
    idxs = density_map > 0
    out = background_bw.copy()
    out[idxs] = colored_density[idxs][:,:3]
    plt.subplot(1, len(sub_species), np.where(sub_species == specie_i)[0][0] + 1)
    plt.title(specie.name)
    plt.imshow(out)
    plt.axis("off")
utils.benchEnd()
plt.tight_layout()
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
plt.savefig(resource_path + "visu/animal_densities.png")

sdlfkj

###############################################################################
# Compute trail weights based on usage, animal count and weight
###############################################################################
utils.benchStart("Computing trail weights")

def getResourceWeight(res_ratio, res_type, specie):
    """
    Resource weight = (ratio resource size / total resource in region) * (need for this resource / total needs)
    """
    request = 0.2
    if res_type == -2:
        request = 0.2
    if res_type >= 0:
        request = (specie.reqs[res_type] / specie.sum_reqs)
    return res_ratio * request
def getTrailWeight(type_a, ratio_a, type_b, ratio_b, region_pops):
    """
    Trail weight = \sum_s Ps(A) * Ps(B) * N_animals(s) * mass(s)
    """
    out = 0
    for specie_i, specie in enumerate(species_herb):
        animal_count = int(region_pops[specie_i])
        if animal_count == 0:
            continue
        weight = getResourceWeight(ratio_a, type_a, specie)
        weight *= getResourceWeight(ratio_b, type_b, specie)
        weight *= animal_count * np.mean(specie.mass)
        out += weight
    return out

for region_i, reg in enumerate(regions):
    weights = {}
    ratios_water = reg.water[:,2] / np.sum(reg.water[:,2])
    local_exits = [(x, y) for x, y in map_exits if accessibility[y, x] == reg.idx]
    for vege_i, ells_i in enumerate(reg.veges):
        if len(ells_i) == 0:
            continue
        areas_i = np.pi * (ells_i[:,2]*.5) * (ells_i[:,3]*.5)
        ratios_i = areas_i / np.sum(areas_i)
        for sub_i in range(len(reg.veges[vege_i])):
            for vege_j in range(vege_i, len(reg.veges)):
                ells_j = reg.veges[vege_j]
                if len(ells_j) == 0:
                    continue
                areas_j = np.pi * (ells_j[:,2]*.5) * (ells_j[:,3]*.5)
                ratios_j = areas_j / np.sum(areas_j)
                start_j = 0 if vege_j != vege_i else sub_i + 1
                for sub_j in range(start_j, len(reg.veges[vege_j])):
                    weight = getTrailWeight(vege_i, ratios_i[sub_i], vege_j, ratios_j[sub_j], region_pop[:,region_i])
                    weight_id = (vege_i, sub_i, vege_j, sub_j)
                    if weight > 0:
                        weights[weight_id] = weight
            for sub_j, ells_j in enumerate(reg.water):
                weight = getTrailWeight(vege_i, ratios_i[sub_i], -1, ratios_water[sub_j], region_pop[:,region_i])
                weight_id = (vege_i, sub_i, -1, sub_j)
                if weight > 0:
                    weights[weight_id] = weight
            for x, y in local_exits:
                weight = getTrailWeight(vege_i, ratios_i[sub_i], -1, 1.0/len(local_exits), region_pop[:,region_i])
                weight_id = (-2, utils.pt2idx(accessibility, (y, x)), vege_i, sub_i)
                if weight > 0:
                    weights[weight_id] = weight
    reg.weights = weights
utils.benchEnd()

###############################################################################
# Compute paths between resources
###############################################################################

def computeShortestPath(reg, qtree, minbounds, maxbounds, path_idx, path_weight):
    xmin, ymin = minbounds
    xmax, ymax = maxbounds
    vege_i, sub_i, vege_j, sub_j = path_idx
    qtree.reset()
    if vege_i == -2:
        sy, sx = utils.idx2pt(accessibility, sub_i)
    else:
        sidx = np.random.choice(reg.veges_px[vege_i][sub_i])
        sy, sx = utils.idx2pt(accessibility, sidx)
    start = qtree.findLeaf(sx - xmin, sy - ymin)
    if start is None:
        print("Error: could not find path:", path_id, path_weight, tgt_out)
        return None
    if vege_j == -1: # Water node
        eidx = np.random.choice(np.arange(len(reg.water_px[sub_j])))
        ex, ey = reg.water_px[sub_j][eidx]
    else:
        eidx = np.random.choice(reg.veges_px[vege_j][sub_j])
        ey, ex = utils.idx2pt(accessibility, eidx)
    end = qtree.findLeaf(ex - xmin, ey - ymin)
    if start.aStar(end):
        segs = []
        curr = end
        total_length = 0
        while curr != start:
            if len(segs) == 0:
                segs.append((curr.x, curr.y, curr.sz))
            segs.append((curr.previous.x, curr.previous.y, curr.previous.sz))
            total_length += np.linalg.norm(curr.center - curr.previous.center)
            curr = curr.previous
        path = {"start":(sx, sy), "end":(int(ex), int(ey)), "weight":path_weight, "length": total_length, "segs":segs}
        return path
    return None

region_paths = []
max_id = int(accessibility.max()) + 1
for region_id in range(max_id):
    py, px = np.where(accessibility == region_id)
    xmin = np.min(px)
    xmax = np.max(px)
    ymin = np.min(py)
    ymax = np.max(py)

    sqszx = 2 << int(np.floor(np.log2(xmax - xmin)))
    sqszy = 2 << int(np.floor(np.log2(ymax - ymin)))
    sqsz = max(sqszx, sqszy)
    im = np.ones((sqsz, sqsz), np.bool)
    im[0:ymax+1-ymin, 0:xmax+1-xmin] = accessibility[ymin:ymax+1, xmin:xmax+1] != region_id

    utils.benchStart("Computing QuadTree for region {}".format(region_id))
    qtree = QTree.QTreeNode(im, im.shape[0])
    utils.benchEnd()

    utils.benchStart("Trails for region {}".format(region_id))
    paths = {}
    reg = regions[region_id]
    reg_weight_items = list(reg.weights.items())
    reg_weight_items.sort(key=lambda x: -x[1])
    total_weight = sum([x[1] for x in reg_weight_items])
    cum_weight = 0
    for path_idx, path_weight in reg_weight_items:
        print("\rPath {}/{}".format(len(paths)+1, len(reg.weights)), end="")
        cum_weight += path_weight
        if cum_weight / total_weight > 0.99:
            break
        path = computeShortestPath(reg, qtree, (xmin, ymin), (xmax, ymax), path_idx, path_weight)
        if path is not None:
            paths[path_idx] = path
    print("")
    region_paths.append({"id": region_id,
                         "qtree": qtree,
                         "min": (xmin, ymin),
                         "max": (xmax, ymax),
                         "paths": paths})
    utils.benchEnd()

def registerEdge(dictionary, v1, v2):
    def registerDirEdge(v1, v2):
        if v1 not in dictionary:
            sub = {}
            dictionary[v1] = sub
        else:
            sub = dictionary[v1]
        if v2 not in sub:
            sub[v2] = 1
        else:
            sub[v2] += 1
    registerDirEdge(v1, v2)
    registerDirEdge(v2, v1)
def getMostUsedEdge(hist):
    idx, count = max(hist.items(), key=lambda x:x[1])
    cx, cy, csz, nx, ny, nsz = idx
    return (cx, cy, csz), (nx, ny, nsz)
def segToIdx(v1, v2):
    cx, cy, csz = v1
    nx, ny, nsz = v2
    item = (cx, cy, csz, nx, ny, nsz)
    if nx < cx or (nx == cx and ny < cy):
        item = (nx, ny, nsz, cx, cy, csz)
    return item
def popEdge(hist, by_vert, v1, v2):
    idx = segToIdx(v1, v2)
    if idx in hist:
        hist.pop(idx)
    by_vert[v1].pop(v2)
    by_vert[v2].pop(v1)
    return idx
def followTrail(hist, by_vert, v1):
    trail = [v1]
    while True:
        items = by_vert[v1].items()
        if len(items) == 0:
            break
        v2 = max(items, key=lambda x:x[1])[0]
        popEdge(hist, by_vert, v1, v2)
        trail.append(v2)
        v1 = v2
    return trail
def cellCenter(cell):
    cx, cy, csz = cell
    return np.array([cx + (csz >> 1), cy + (csz >> 1)])
def prolongTrail(trails, first_appearance, t0, t1):
    """
    Find extension of trail starting/ending at t0->t1 in previously extracted trails
    Extension with lowest angle is kept
    """
    v1 = cellCenter(t0)
    v2 = cellCenter(t1)
    seg = (v2 - v1) / np.linalg.norm(v2 - v1)
    first_app = first_appearance[t0]
    orig_trail = trails[first_app[0]]
    candidates = []
    if first_app[1] > 0:
        candidates.append(orig_trail[first_app[1] - 1])
    if first_app[1] < len(orig_trail) - 1:
        candidates.append(orig_trail[first_app[1] + 1])
    cand_centers = [cellCenter(candidate) for candidate in candidates]
    cand_segs = [(v1 - v) / np.linalg.norm(v1 - v) for v in cand_centers]
    cand_dot = [np.dot(cand_seg, seg) for cand_seg in cand_segs]
    return candidates[np.argmax(cand_dot)]

###############################################################################
# Refine paths into trails
###############################################################################
utils.benchStart("Refining trails")
region_weights = region_pop.sum(0)
region_weights = region_weights / region_weights.max()
trail_map = np.zeros(tuple(np.array(accessibility.shape) // bg_scale), dtype=np.uint8)
for region_item in region_paths:
    region_id = region_item["id"]
    xmin, ymin = region_item["min"]
    xmax, ymax = region_item["max"]
    paths = region_item["paths"].values()
    hist = {}
    by_vert = {}
    for i,path_obj in enumerate(paths):
        segs = path_obj["segs"]
        weight = path_obj["weight"]
        for j in range(len(segs) - 1):
            v1 = segs[j]
            v2 = segs[j+1]
            registerEdge(by_vert, v1, v2)
            item = segToIdx(v1, v2)
            if item not in hist:
                hist[item] = weight
            else:
                hist[item] += weight
    if len(hist.values()) == 0:
        continue
    weights_hist = hist.copy() # Make a copy since hist will be emptied
    trails = []
    first_appearance = {}
    to_draw = []
    while len(hist):
        v1, v2 = getMostUsedEdge(hist)
        start = popEdge(hist, by_vert, v1, v2)
        trail = followTrail(hist, by_vert, v1)[::-1]
        trail += followTrail(hist, by_vert, v2)
        # Extend trail to previously instanciated ones for interpolation
        if len(trail) > 1 and trail[0] in first_appearance:
            trail = [prolongTrail(trails, first_appearance, trail[0], trail[1])] + trail
        if len(trail) > 1 and trail[-1] in first_appearance:
            trail = trail + [prolongTrail(trails, first_appearance, trail[-1], trail[-2])]
        # Register first appearance of vertices to connect subpaths
        for i, item in enumerate(trail):
            if item not in first_appearance:
                first_appearance[item] = (len(trails), i)
        trails.append(trail)
        xs = np.array([cx + (csz >> 1) for cx, cy, csz in trail])
        ys = np.array([cy + (csz >> 1) for cx, cy, csz in trail])
        # Extract/interpolate weights
        wx = []
        wy = []
        maxusage = max(weights_hist.values())
        seglens = np.sqrt((xs[1:]-xs[:-1])**2 + (ys[1:]-ys[:-1])**2)
        cumlens = np.cumsum(seglens)
        currlen = 0
        for i in range(len(trail)-1):
            usage = weights_hist[segToIdx(trail[i], trail[i+1])]
            weight = usage / maxusage
            if i == 0:
                wx.append(0)
                wy.append(weight)
            wx.append(currlen + seglens[i] * 0.5)
            wy.append(weight)
            if i == len(trail)-2:
                wx.append(currlen + seglens[i])
                wy.append(weight)
            currlen += seglens[i]
        subdiv = int(max((wx[-1] - wx[0]) / 5, 5))
        wx2 = np.linspace(wx[0], wx[-1], subdiv)
        wx2 = wx2[:-1] + (wx2[1]-wx2[0]) * .5
        spl = interpolate.interp1d(wx, wy)
        wy2 = spl(wx2)
        if len(xs) > 3:
            fint, u = interpolate.splprep([xs, ys], s=0)
            xint, yint = interpolate.splev(np.linspace(0, 1, subdiv), fint)
        else:
            xint = xs
            yint = ys
        for i in range(len(xint) - 1):
            lx1 = int(xint[i] + xmin)
            lx2 = int(xint[i+1] + xmin)
            ly1 = int(yint[i] + ymin)
            ly2 = int(yint[i+1] + ymin)
            th = int(10 * (wy2[i]+1))
            col = int(255 * (wy2[i] ** .9))
            if wy2[i] < 0.01:
                col = 0
            to_draw.append(((lx1, ly1), (lx2, ly2), col, th))
    to_draw.sort(key=lambda x: x[2])
    for p1, p2, col, th in to_draw:
        if th > 0 and col > 0:
            p1 = tuple(np.array(p1) // bg_scale)
            p2 = tuple(np.array(p2) // bg_scale)
            cv2.line(trail_map, p1, p2, col, thickness=th // bg_scale)
alpha = trail_map / 256.0
alpha = np.dstack((alpha, alpha, alpha))
trails_full = background * (1.0 - alpha) + 1.0 * alpha
skimage.io.imsave(resource_path + 'visu/trails.png', np.uint8(trails_full * 256))
utils.benchEnd()

###############################################################################
# Compute daily plannings
###############################################################################
utils.benchStart("Computing daily plannings")

def getRandomResource(specie, reg):
    cs = np.cumsum(np.array(specie.reqs) / specie.sum_reqs)
    res_type = np.sum(np.random.random() >= cs)
    if res_type < len(reg.veges_px):
        cs_resource = np.cumsum([len(x) for x in reg.veges_px[res_type]])
        res_id = np.sum((np.random.random() * cs_resource[-1]) >= cs_resource)
    else:
        res_type = -2
        res_id = 0
    return res_type, res_id
def sortedPathIdx(vg_i, sub_i, vg_j, sub_j):
    if vg_i == -1 or (vg_j < vg_i and vg_j >= 0) or (vg_i == vg_j and sub_j < sub_i):
        return (vg_j, sub_j, vg_i, sub_i)
    return (vg_i, sub_i, vg_j, sub_j)
def getRegionPath(reg, vg_i, sub_i, vg_j, sub_j):
    regpaths = region_paths[reg.idx]
    paths = regpaths["paths"]
    path_idx = sortedPathIdx(vg_i, sub_i, vg_j, sub_j)
    if path_idx in paths:
        return paths[path_idx]
    # If the path was not already computed, compute it now
    path = computeShortestPath(reg, regpaths["qtree"], regpaths["min"], regpaths["max"], path_idx, 0.0)
    if path is not None:
        paths[path_idx] = path
        return path
    return None
def getFullPlan(specie, reg, plan):
    out = []
    total = 0
    for i in range(len(plan)):
        p1 = plan[i]
        out.append({"type":int(p1[0]), "nodeidx": int(p1[1]), "duration": int(p1[2])})
        total += p1[2]
        if i < len(plan) - 1:
            p2 = plan[i + 1]
            if p1[0] == -2 or p2[0] == -2:
                out.append({"type":-1, "nodeidx":-1, "duration": 30})
            else:
                path = getRegionPath(reg, p1[0], p1[1], p2[0], p2[1])
                if path is not None:
                    length = path["length"]
                    time = 60 * length / (specie.walk_speed*1000)
                    total += time
                    out.append({"type":-1, "nodeidx":-1, "duration": int(time)})
    return out
herds_by_id = []
for sp_i, sp in enumerate(region_herds):
    for reg_i, herds in enumerate(sp):
        for herd in herds:
            herds_by_id.append((reg_i, sp_i, herd))
for sp_i, sp in enumerate(carn_herds):
    for reg_i, herds in enumerate(sp):
        for herd in herds:
            herds_by_id.append((reg_i, sp_i + len(species_herb), herd))
np.random.shuffle(herds_by_id)
plans_obj = []
for reg_i, sp_i, count in herds_by_id:
    if sp_i < len(species_herb):
        specie = species_herb[sp_i]
    else:
        specie = species_carn[sp_i - len(species_herb)]
    reg = regions[reg_i]
    full_duration = 0
    plan = []
    water_cs = np.cumsum(reg.water[:,2] / np.sum(reg.water[:,2]))
    for i in range(specie.water_req):
        if i > 0:
            res_type, res_id = getRandomResource(specie, reg)
            duration = np.random.rand() * 60 + 90
            full_duration += duration
            plan.append((res_type, res_id, duration))
        duration = np.random.rand() * 60 + 90
        full_duration += duration
        plan.append((-1, np.sum(np.random.random() >= water_cs), duration))
    while full_duration / 60 < 18:
        res_type, res_id = getRandomResource(specie, reg)
        duration = np.random.rand() * 60 + 90
        full_duration += duration
        plan.insert(np.random.randint(len(plan)+1), (res_type, res_id, duration))
    full_plan = getFullPlan(specie, reg, plan)
    plans_obj.append({"sp_i": sp_i, "reg_i": reg_i, "count": count, "plan": full_plan})
utils.benchEnd()
