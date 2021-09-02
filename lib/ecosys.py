import numpy as np

# convert data into a [1 - 10 [ scale
def convert(data, value_1, value_9):

    a = 8/(value_9-value_1)
    b = 1 - a * value_1

    return a * data + b


def compute_moisture(drain_map, monthly_precip):

    result = np.zeros((12, drain_map.shape[0],  drain_map.shape[0]), dtype = drain_map.dtype)

    for i, p in enumerate(monthly_precip):
        result[i] = drain_map + np.log(p)

    return result


def monthly_vigor(low, high, data0, data1):
    """
    Return vigor map between month1 and month2
    """

    mx = np.maximum(data0, data1)
    mn = np.minimum(data0, data1)

    diff = mx - mn
    same_mask = diff == 0
    diff[same_mask] = 1

    max_in = np.minimum(high, mx)
    min_in = np.maximum(low, mn)

    size_in = np.maximum(0, max_in - min_in)

    vigor = size_in / diff

    vigor[same_mask] = np.logical_and(mx >= low, mx <= high)[same_mask]
    vigor[same_mask] = np.logical_and(mx >= low, mx <= high)[same_mask]
    return vigor


def yearly_vigor(low, high, year_data):

    result = np.zeros_like(year_data[0])

    for i in range(11):
        result += monthly_vigor(low, high, year_data[i], year_data[i+1])

    result += monthly_vigor(low, high, year_data[11], year_data[0])

    return result / 12

def geology_vigor(plant_geol, ground_map):

    result = np.zeros(ground_map.shape, dtype = np.float64)
    for i, v in enumerate(plant_geol):
        result += (ground_map == i) * v
    return result

def plant_density(plant_info, condition_list, geology = None):

    density = np.ones_like(condition_list.values().__iter__().__next__()[0])

    for name, v in plant_info.items():
        if name in condition_list:
            density = np.minimum(density, yearly_vigor(v[0], v[1], condition_list[name]))

    if "geology" in plant_info and geology is not None:
        geo_vigor = geology_vigor(plant_info["geology"], geology)
        density = np.minimum(density, geo_vigor)

    return density

def plant_compet(plants, condition_list, geology):

    # copy conditions
    cond = {}
    for name, t in condition_list.items():
        cond[name] = t.copy()

    result = np.zeros_like(condition_list.values().__iter__().__next__()[0]) +  np.zeros(len(plants))[:, np.newaxis, np.newaxis]

    #compute vigor
    gpi = {}
    for i, pi in enumerate(plants):
        result[i] = plant_density(pi, cond, geology)
        for name, v in pi.items():
            if name in condition_list:
                if not name in gpi:
                    gpi[name] = np.zeros((3, len(plants)))
                gpi[name][:, i] = v
            elif name == "geology":
                if not name in gpi:
                    gpi[name] = np.zeros((len(v), len(plants)))
                gpi[name][:, i] = v

    #locally sort plants by vigor
    parse = np.argsort(result, axis = 0)
    result_sorted = np.zeros_like(result)

    for i, p in enumerate(parse):

        #create conditions
        new_pi = {}
        for name, v in gpi.items():
            if name in condition_list or name == "geology":
                new_pi[name] = v[:, p]

        #recompute vigor
        result_sorted[i] = plant_density(new_pi, cond, geology)

        #update conditions
        for name, v in gpi.items():
            if name in cond:
                cond[name] -= v[2, p] * result_sorted[i]

    np.put_along_axis(result, parse, result_sorted, 0)

    return result
