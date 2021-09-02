import numpy as np
import skimage
import numba

default_colors = ("ccccfd", "0000ff", "52d252", "adadad", "ffffff" )

def load(filename, colors = default_colors):
    rgb_data = skimage.io.imread(filename).astype(np.intp)
    int_data = rgb_data[:, :, 0] * 256 * 256 + rgb_data[:, :, 1] * 256 +  rgb_data[:, :, 2]

    # clean data
    npcolors = np.array([int(c, 16) for c in colors], dtype = rgb_data.dtype)
    out = np.empty_like(int_data)
    out[:,:] = -1
    ne = 1
    while ne != 0:
        pe, ne = clean(int_data, out, npcolors)
        if ne == int_data.size:
            assert False, "The color of the image seems to be wrong"
    return out

@numba.njit
def clean(data, out, npcolors):
    """
    Remove all erroneous colors, replaced by the most commonly found in the direct neighborhood
    """
    prev_err = 0
    new_err = 0

    old = data.copy()

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):

            found = -1
            for i, col in enumerate(npcolors):
                if data[r, c] == col:
                    found = i

            if found == -1:
                prev_err += 1
                count = np.zeros(npcolors.shape[0], dtype = np.intp)

                for ir in range(max(r-1, 0), min(r+2, data.shape[0])):
                    for ic in range(max(c-1, 0), min(c+2, data.shape[1])):
                        for i, col in enumerate(npcolors):
                            if old[ir, ic] == col:
                                count[i] += 1

                ic = np.argmax(count)
                if count[ic] != 0:
                    data[r, c] = npcolors[ic]
                    found = ic
                else:
                    new_err += 1

            if found != -1:
                out[r, c] = found
    return prev_err, new_err
