import os
import sys
import scipy
import math
import skimage
import skimage.io
import numpy as np
sys.path.append("lib")
import explicitdiffusion
import drain
import utils
# Illumination can be computed on the GPU if you have cuda installed
# In this case, import sun_cuda instead of sun
import sun
# import sun_cuda as sun

resource_path = "resources/raw/"
dem = skimage.io.imread(resource_path + "dem_filled.tif")

if not os.path.exists(resource_path + "water.tif"):
    utils.benchStart("Computing water")
    source = np.ones_like(dem)
    source[1,1196:1206] = 305 * 1000 * 1000 / (1206 - 1196)
    area = drain.compute(dem, source, fill = False)
    gx, gy = np.gradient(dem)
    gn = np.sqrt(gx*gx+gy*gy)
    correct = (np.abs(gx) + np.abs(gy))
    area_rot = area * gn / correct
    water = (area_rot / (.1 + gn**.5))**(2/3)
    water /= np.max(water)
    slope_diffusivity = np.exp(math.log(.01) / .57*gn)

    def run(water):
        diff = water.copy()
        for i in range(500):
            # diffusivity: depends on h and inversed gradient
            d = diff * slope_diffusivity
            diff, dt = explicitdiffusion.solve_step(d, diff, 1, 1000)
        return diff

    diff = run(water)
    diff = diff/np.max(diff)
    skimage.io.imsave(resource_path + 'water.tif', diff.astype(np.float32))
    utils.benchEnd()

for month in range(0,12):
    illum_fname = resource_path + 'illum-'+str(month)+'.tif'
    if not os.path.exists(illum_fname):
        utils.benchStart("Computing illumination (%d/%d)" % (month+1, 12))
        illum = sun.illumination(dem, 1, 42.85, month)
        skimage.io.imsave(illum_fname, illum.astype(np.float32))
        utils.benchEnd()

# temperature: 0 degrees at sea level, increase with altitude
if not os.path.exists(resource_path + "temp.tif"):
    utils.benchStart("Computing temperature")
    temperature = 0 - dem /100
    skimage.io.imsave(resource_path + "temp.tif", temperature)
    utils.benchEnd()
