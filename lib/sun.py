import numpy as np
import math
import numba

def illumination(dem, cell_size, latitude, month):
    illum = np.zeros_like(dem)
    max_illum = 0
    for minutes in range(0, 1440, 60):
        max_illum += 1.0
        sundir = sun_direction(latitude, month, minutes)
        if sundir[1] > 0.0:
            exposure_all(dem, cell_size, sundir, illum)
    return illum / max_illum

def sun_direction(latitude, month, minutes, north_guess = [0, 0, -1]):
    """
    north_guess should give the xz coordinate of the "compass" north (with a 0 as y)
    """
    _axis_tilt = 0.408407
    _monthly_axis_tilt = _axis_tilt/3.0
    _half_day_in_minutes = 720.0

    def rotate(U, N, angle):
        cosangle = math.cos(angle)
        return cosangle*U + ((1.0-cosangle)*np.dot(U,N))*N + math.sin(angle) * np.cross(N, U)

    # Frame centered at (noon, latitude)

    # Earth normal (0, 1, 0)
    # correct the north vector so that it is orthogonal to east and earth normal

    east = rotate(np.array(north_guess), np.array([0.0,1.0,0.0]), -.5 * math.pi)
    north = rotate(np.array(north_guess), east, latitude / 180.0 * math.pi)

    # earth axis tilt
    max_axis_tilt = -_axis_tilt + (abs(6.0 - month) * _monthly_axis_tilt)

    # day angle
    day_angle = (( _half_day_in_minutes - minutes) / _half_day_in_minutes) * math.pi

    #midday, equinox
    sun = np.cross(east, north)
    sun = rotate(sun, east, max_axis_tilt)
    sun =  rotate(sun, north, day_angle)
    return sun / math.sqrt(np.dot(sun, sun))

@numba.njit(parallel=True)
def exposure_all(dem, cell_size, sundir, illum):
    for r in numba.prange(dem.shape[0]):
        for c in range(dem.shape[1]):
            illum[r, c] += exposure(dem, cell_size, sundir, r, c)

@numba.njit
def exposure(dem, cell_size, sundir, r, c):
    t = 0
    ray_start_h = dem[r, c] / cell_size

    illum = 1.0

    extent_r = np.float64(dem.shape[0])
    extent_c = np.float64(dem.shape[1])
    extent = np.int64(math.ceil(math.sqrt(extent_r*extent_r + extent_c*extent_c)))

    dt = 1

    ray_l = math.sqrt(sundir[0]*sundir[0]+sundir[2]*sundir[2])
    ray_dc = sundir[0] / ray_l
    ray_dh = sundir[1] / ray_l
    ray_dr = sundir[2] / ray_l

    #compute soft shadow
    for i in range(1024):
        t+= dt
        dt*= (10.36)**(1/1024)

        ray_c = t*ray_dc
        ray_h = t*ray_dh
        ray_r = t*ray_dr

        ri = np.int64(math.floor(r + ray_r))
        ci = np.int64(math.floor(c + ray_c))

        if ri>=dem.shape[0] or ci >=dem.shape[1] or ri<0 or ci<0:
            break

        if ri == r and ci == c:
            continue

        #elevation of the terrain at the sampled location
        terrain_h = dem[ri, ci] / cell_size

        #direction to the terrain
        terrain_dir_h = terrain_h - ray_start_h
        terrain_norm = math.sqrt(ray_c*ray_c + terrain_dir_h*terrain_dir_h + ray_r*ray_r)
        terrain_dir_c = ray_c / terrain_norm
        terrain_dir_h = terrain_dir_h / terrain_norm
        terrain_dir_r = ray_r / terrain_norm

        #compute proportion of visible sun
        sun_amount = min(1.0, max(0.0, math.acos(terrain_dir_c*sundir[0] + terrain_dir_h * sundir[1] + terrain_dir_r * sundir[2]) / 0.1309 ))*0.5

        if terrain_h > ray_start_h + ray_h:
            sun_amount = 0.5 - sun_amount
        else:
            sun_amount = 0.5 + sun_amount

        illum = min(illum, sun_amount)

        if illum < 0.00001:
            break

    #compute local illumination
    cm = dem[r, max(c-1, 0)]
    cp = dem[r, min(c+1, dem.shape[1]-1)]
    rm = dem[max(r-1, 0), c]
    rp = dem[min(r+1, dem.shape[0]-1), c]

    gr = rm-rp
    gc = cm-cp
    gh = 2.0 * cell_size

    l = math.sqrt(gc*gc + gh*gh + gr*gr)

    dp = (gc * sundir[0] + gh * sundir[1] + gr * sundir[2] ) / l

    return illum * max(0.0, dp)
