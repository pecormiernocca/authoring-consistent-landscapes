import numpy as np
from heapq import *
from enum import *

class QTreeNode:
    class Dir(IntEnum):
        NW = 0
        NE = 1
        SW = 2
        SE = 3
        N = 4
        S = 5
        W = 6
        E = 7

    def __init__(self, im, sz, x = 0, y = 0, parent = None):
        self.parent = parent
        self.im = im
        self.x = x
        self.y = y
        self.yx = (y, x)
        self.center = np.array([x, y]) + (sz >> 1)
        self.sz = sz
        self.empty = True
        self.neighbors = None
        self.previous = None
        self.cost = np.inf
        self.__subdiv()

    def __subdiv(self):
        self.children = []
        y, x = self.yx
        sz = self.sz
        roi = self.im[y:y+sz, x:x+sz]
        if True not in roi:
            return
        if False not in roi:
            self.empty = False
            return
        nsz = sz >> 1
        # 0 1
        # 2 3
        self.children.append(QTreeNode(self.im, nsz, x, y, self))
        self.children.append(QTreeNode(self.im, nsz, x + nsz, y, self))
        self.children.append(QTreeNode(self.im, nsz, x, y + nsz, self))
        self.children.append(QTreeNode(self.im, nsz, x + nsz, y + nsz, self))

    def draw(self, canvas):
        col = (1, 0, 0)
        if self.empty:
            col = (0, 1, 0)
        y, x = self.yx
        sz = self.sz
        canvas[y:y+sz, x:x+sz] = col

    def drawLeafs(self, canvas):
        if len(self.children):
            for child in self.children:
                child.drawLeafs(canvas)
            return
        col = (1, 0, 0)
        if self.empty:
            col = (0, 1, 0)
        y, x = self.yx
        sz = self.sz
        canvas[y:y+sz, x] = col
        canvas[y:y+sz, x+sz-1] = col
        canvas[y, x:x+sz] = col
        canvas[y+sz-1, x:x+sz] = col

    def __neighborGreaterEqual(self, direction):
        if self.parent is None:
            return None
        siblings = self.parent.children
        if direction == self.Dir.N:
            if siblings[self.Dir.SW] == self:
                return siblings[self.Dir.NW]
            if siblings[self.Dir.SE] == self:
                return siblings[self.Dir.NE]
        elif direction == self.Dir.S:
            if siblings[self.Dir.NW] == self:
                return siblings[self.Dir.SW]
            if siblings[self.Dir.NE] == self:
                return siblings[self.Dir.SE]
        elif direction == self.Dir.W:
            if siblings[self.Dir.NE] == self:
                return siblings[self.Dir.NW]
            if siblings[self.Dir.SE] == self:
                return siblings[self.Dir.SW]
        else:
            if siblings[self.Dir.NW] == self:
                return siblings[self.Dir.NE]
            if siblings[self.Dir.SW] == self:
                return siblings[self.Dir.SE]

        node = self.parent.__neighborGreaterEqual(direction)
        if node is None or len(node.children) == 0:
            return node

        if direction == self.Dir.N:
            return node.children[self.Dir.SW if siblings[self.Dir.NW] == self else self.Dir.SE]
        elif direction == self.Dir.S:
            return node.children[self.Dir.NW if siblings[self.Dir.SW] == self else self.Dir.NE]
        elif direction == self.Dir.W:
            return node.children[self.Dir.NE if siblings[self.Dir.NW] == self else self.Dir.SE]
        else:
            return node.children[self.Dir.NW if siblings[self.Dir.NE] == self else self.Dir.SW]

    def __neighborsSmaller(self, neighbor, direction):
        candidates = []
        if neighbor:
            candidates.append(neighbor)
        neighbors = []

        while len(candidates):
            curr = candidates.pop()
            if len(curr.children) == 0:
                neighbors.append(curr)
                continue
            if direction == self.Dir.N:
                candidates.append(curr.children[self.Dir.SW])
                candidates.append(curr.children[self.Dir.SE])
            elif direction == self.Dir.S:
                candidates.append(curr.children[self.Dir.NW])
                candidates.append(curr.children[self.Dir.NE])
            elif direction == self.Dir.W:
                candidates.append(curr.children[self.Dir.NE])
                candidates.append(curr.children[self.Dir.SE])
            else:
                candidates.append(curr.children[self.Dir.NW])
                candidates.append(curr.children[self.Dir.SW])
        return neighbors

    def getNeighbors(self):
        if self.neighbors is None:
            self.neighbors = []
            for direction in [self.Dir.N, self.Dir.S, self.Dir.W, self.Dir.E]:
                neighb = self.__neighborGreaterEqual(direction)
                smaller = self.__neighborsSmaller(neighb, direction)
                self.neighbors += [item for item in smaller if item.empty]
        return self.neighbors

    def findLeaf(self, x, y):
        if not self.x <= x < self.x+self.sz or not self.y <= y < self.y+self.sz:
            return None
        if len(self.children) == 0:
            return self
        for child in self.children:
            ret = child.findLeaf(x, y)
            if ret:
                return ret
        return None

    def count(self, count_empty = True):
        if len(self.children) == 0:
            return self.empty == count_empty
        return sum([child.count(count_empty) for child in self.children])

    def reset(self):
        self.previous = None
        self.cost = np.inf
        for child in self.children:
            child.reset()

    def aStar(self, end):
        if not self.empty or not end.empty:
            return False
        self.cost = 0.0
        frontier = [(0.0, self)]
        while len(frontier):
            item = heappop(frontier)
            cost, node = item
            if node == end:
                return True
            for neighb in node.getNeighbors():
                new_cost = cost + np.linalg.norm(node.center - neighb.center)
                if new_cost < neighb.cost:
                    neighb.cost = new_cost
                    neighb.previous = node
                    priority = new_cost + np.linalg.norm(neighb.center - end.center)
                    heappush(frontier, (priority, neighb))
        return False

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.sz < other.sz
        return NotImplemented

