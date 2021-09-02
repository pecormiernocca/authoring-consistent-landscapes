import numpy as np

class AnimalSpecies(object):
    def __init__(self, name, group_sz, mass, walk_speed, length, surplus = 0,
                 water_req = 2):
        """
        water_req:
            Number of times that access to water is needed per day
        """
        self.name = name
        self.group_size = group_sz
        self.mass = mass
        self.walk_speed = walk_speed
        self.length = length
        self.surplus = surplus
        self.water_req = water_req

class HerbivoreSpecies(AnimalSpecies):
    def __init__(self, name, group_sz, mass, walk_speed, length, surplus,
                 water_req, plant_req = 0, shrub_req = 0, tree_req = 0):
        """
        plant_req, shrub_req:
            Number of square meters of grass/shrubs nedded per day
        """
        super().__init__(name, group_sz, mass, walk_speed, length, surplus,
                         water_req)
        self.plant_req = plant_req
        self.shrub_req = shrub_req
        self.tree_req = tree_req
        self.reqs = [shrub_req, plant_req, tree_req]
        self.sum_reqs = sum(self.reqs)

class CarnSpecies(HerbivoreSpecies):
    def __init__(self, name, group_sz, mass, walk_speed, length,
                 water_req, plant_req = 0, shrub_req = 0, tree_req = 0,
                 herbivores_req = 0):
        super().__init__(name, group_sz, mass, walk_speed, length, 0,
                         water_req, plant_req, shrub_req, tree_req)
        self.herbivores_req = herbivores_req
        self.reqs += [herbivores_req]
        self.sum_reqs = sum(self.reqs)

# If a species needs 1m²/day of grass
# they will need YEARLY_RATIO m²/year when taking into account regrowth
# This currently doesn't account for faster regrowth depending on temperature and humidity
# Should be around 1000/4 and 5000/4
YEARLY_RATIO = 750.0

species_herb = [
    HerbivoreSpecies("Bison", (10, 35), (700, 1000), 5.0, (2.1, 3.8), 0.1,
                  2, 25, 2.7),
    HerbivoreSpecies("Elk", (30, 45), (70, 250), 5.0, (1.6, 2.7), 0.2,
                  2, 18, 0),
    HerbivoreSpecies("Deer", (5, 14), (30, 80), 5.0, (1.3, 1.75), 0.2,
                  2, 6, 0),
    HerbivoreSpecies("Reindeer", (50, 100), (100, 300), 6.0, (1.5, 2.3), 0.2,
                  2, 15, 10, 5),
    HerbivoreSpecies("Mouflon", (20, 150), (130, 185), 5.0, (1.6, 1.9), 0.75,
                  2, 13, 0, 0),
    HerbivoreSpecies("Rhino", (1, 2), (1500, 2000), 5.0, (3.6, 3.8), 0.5,
                  2, 30, 10, 0),
    HerbivoreSpecies("Horse", (6, 20), (227, 900), 7.0, (2.2, 2.8), 0.75,
                  2, 26, 0, 0)
]
species_carn = [
    CarnSpecies("Bear", (1, 2), (200, 600), 5.0, (1.0, 2.8),
                  2, 20, 20, 0, 5),
    CarnSpecies("Wolf", (5, 9), (40, 80), 8.0, (1.0, 1.3),
                  2, 0, 0, 0, 9),
    CarnSpecies("Lion", (2, 30), (126, 272), 7.0, (2.4, 3.3),
                  2, 0, 0, 0, 27),
    CarnSpecies("Lynx", (1, 2), (11, 15), 7.0, (0.8, 1.3),
                  2, 0, 0, 0, 2),
    CarnSpecies("Fox", (1, 4), (3, 14), 6.0, (0.4, 0.9),
                  2, 0, 1, 0, 1)
]

def fitness(regions, animal, grp_sz, region_surplus=None, cell_width_sq = 1):
    shrub_req = animal.shrub_req * (YEARLY_RATIO / cell_width_sq) * grp_sz
    plant_req = animal.plant_req * (YEARLY_RATIO / cell_width_sq) * grp_sz
    tree_req = animal.tree_req * (YEARLY_RATIO / cell_width_sq) * grp_sz
    reqs = [shrub_req, plant_req, tree_req]
    out = []
    for reg_i, region in enumerate(regions):
        fit = np.inf
        for i in range(3):
            avail = region.resources[i]
            if reqs[i] == 0:
                curr_fit = np.inf
            else:
                curr_fit = avail / reqs[i]
            fit = min(fit, curr_fit)
        if region_surplus is not None:
            avail = region_surplus[:,reg_i].sum()
            req = animal.herbivores_req * grp_sz
            if req == 0:
                curr_fit = np.inf
            else:
                curr_fit = avail / req
            fit = min(fit, curr_fit)
        water = region.water[:,2].sum()
        fit = min(fit, water / (grp_sz * 2))
        out.append(fit)
    return np.array(out)

def getCandidateId(instance_count, total, tgt_density, species_used=species_herb):
    div = np.maximum(total, 1)
    grps = [np.random.randint(*animal.group_size) for animal in species_used]
    current = ((instance_count / div) - tgt_density) ** 2
    base = ((instance_count) / (div+grps) - tgt_density) ** 2
    addon = ((instance_count+grps) / (div+grps) - tgt_density) ** 2
    errs = np.sum(base) - base + addon
    curr_err = np.sum(current)
    idx = np.argmax(curr_err - errs)
    return idx, grps[idx]
