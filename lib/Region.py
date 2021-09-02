import numpy as np
import numba
import utils

class Region:
    def __init__(self, idx, veges, veges_px, water, water_px):
        """
        veges: list of list of ellipses
        """
        self.veges = veges
        self.veges_px = veges_px
        self.water = water
        self.water_px = water_px
        self.idx = idx
        self.resources = []
        self.resetMaxResources()

    def resetMaxResources(self):
        self.resources = [self.getMaxResources(i) for i in range(len(self.veges))]

    def getMaxResources(self, i):
        ells = self.veges[i]
        if len(ells) == 0:
            return 0
        return np.sum(np.pi * (ells[:,2]*.5) * (ells[:,3]*.5))

    def plot(self, ax):
        for i in range(len(self.veges)):
            utils.pltEllipses(ax, self.veges[i], presence_colors[i])
            ax.scatter(self.veges[i][:,0], self.veges[i][:,1], c=presence_colors[i])
        utils.pltEllipses(ax, self.water, "blue", "none")

    def toObj(self):
        veges_list = [{"nodes":[{"pxs":j.tolist()} for j in i]} for i in self.veges_px]
        water_list = [{"pxs":np.sum(i * (1, accessibility.shape[0]), axis=1).tolist()} for i in self.water_px]
        obj = {"veges_px": veges_list,
               "water_px": water_list,
               "idx": self.idx}
        return obj

