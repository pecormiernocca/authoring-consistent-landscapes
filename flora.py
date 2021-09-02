import matplotlib.pyplot as plt
import matplotlib.colors as mcols
import numpy as np
import skimage
import time
import sys
import pathlib

sys.path.append("./lib")
import ecosys_param
import geology
import ecosys
import utils

utils.benchStart("Loading data")

sol = "J"
resource_path = "resources/tautavel_%s/" % sol
pathlib.Path(resource_path + "flora").mkdir(parents=True, exist_ok=True)

# load exposure
illum  = skimage.io.imread(resource_path + '../raw/illum-0.tif') + np.zeros(12)[:, np.newaxis, np.newaxis]
for i in range(1, 12):
    illum[i] = skimage.io.imread(resource_path + '../raw/illum-' + str(i) + '.tif')

# load temperature
temp0 = skimage.io.imread(resource_path + '../raw/temp.tif') + ecosys_param.offset_temperature

# load drainage
drain = np.log(skimage.io.imread(resource_path + '../raw/water.tif')) + ecosys_param.offset_ground_humidity

# load geology
geol = geology.load(resource_path + "../raw/soil.png", colors = ecosys_param.geologie_colors)

monthly_temp = ecosys.convert(temp0 + ecosys_param.monthly_avg_temp[:, np.newaxis, np.newaxis],
                              ecosys_param.calibration_temperature[0],
                              ecosys_param.calibration_temperature[1])
monthly_drain = ecosys.convert(drain + np.log(ecosys_param.monthly_precip)[:, np.newaxis, np.newaxis],
                               ecosys_param.calibration_ground_humidity[0],
                               ecosys_param.calibration_ground_humidity[1])
monthly_illum = ecosys.convert(illum,
                               ecosys_param.calibration_illum[0],
                               ecosys_param.calibration_illum[1])

conditions  = {'illum' : monthly_illum, 'temp' : monthly_temp, 'moisture' : monthly_drain}
utils.benchEnd()

utils.benchStart("Running competition")
plnts = ecosys.plant_compet(ecosys_param.plants, conditions, geol)
utils.benchEnd()

utils.benchStart("Exporting")
for i in range(plnts.shape[0]):
    plnt = plnts[i]
    skimage.io.imsave(resource_path + 'flora/plant-' + str(i) + '.tif', plnt.astype(np.float16))
utils.benchEnd()

