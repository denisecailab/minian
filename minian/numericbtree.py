import numpy as np
from intervaltree import Interval, IntervalTree
from collections import OrderedDict
from lrange import lrange


def interval_intersect(self, other):
    if self.overlaps(other):
        return Interval(max(self.begin, other.begin), min(self.end, other.end))


def interval_subtract(self, other):
    result = []
    if self.contains_interval(other):
        result.append(Interval(self.begin, other.begin))
        result.append(Interval(other.end, self.end))
        return set(filter(lambda it: not it.is_null(), result))


def interval_toset(self):
    return set(list(range(self.begin, self.end)))


def interval_iter(self):
    for n in lrange(self.begin, self.end):
        yield n

Interval.intersect = interval_intersect
Interval.subtract = interval_subtract
Interval.to_set = interval_toset
Interval.to_iter = interval_iter


def intervaltree_intersect(self, other):
    result = set()
    for itvl1 in self:
        for itvl2 in other.search(itvl1.begin, itvl1.end):
            result.add(itvl1.intersect(itvl2))
    return IntervalTree(result)


def intervaltree_subtract(self, other):
    for itvl in other:
        self.chop(itvl.begin, itvl.end)


def intervaltree_toset(self):
    result = set()
    for itvl in self:
        result.update(itvl.to_set())
    return result


def intervaltree_iter(self):
    for itvl in sorted(self):
        for n in itvl.to_iter():
            yield n


IntervalTree.intersect = intervaltree_intersect
IntervalTree.subtract = intervaltree_subtract
IntervalTree.to_set = intervaltree_toset
IntervalTree.to_iter = intervaltree_iter


class NBTree(object):
    def __init__(self, depth, dview=None, nodes=None):
        self._depth = depth
        self._dview = dview
        if not nodes:
            self._nodes = OrderedDict(
                (i, IntervalTree.from_tuples([(2**i - 1, 2**(i+1) - 1)])) for i in range(1, depth + 1))
        else:
            self._nodes = nodes

    def get_nodes(self, level=None, unpack=False):
        if level:
            try:
                for lev in level:
                    if unpack:
                        for n in self._nodes[lev].to_iter():
                            yield n
                    else:
                        yield self._nodes[lev]
            except TypeError:
                if unpack:
                    for n in self._nodes[level].to_iter():
                        yield n
                else:
                    yield self._nodes[level]
        else:
            for inttree in self._nodes.items():
                if unpack:
                    for n in inttree[1].to_iter():
                        yield n
                else:
                    yield inttree

    def in_level(self, level, nid, check=True):
        if check:
            return self._nodes[level].overlaps(nid)
        else:
            return 2**level - 1 <= nid < 2**(level+1) - 1

    def level(self, nid, check=True):
        for lev in range(1, self._depth + 1):
            if self.in_level(lev, nid, check):
                return lev
        else:
            return 0

    def leaves(self, nid, level=None):
        if not level:
            level = self._depth
        lb = 2**level - 1
        hb = 2**(level+1) - 2
        levn = hb-lb + 1
        levid = self.level(nid)
        shift = nid - (2**levid - 1)
        div = 2**levid
        llea = lb + levn/div*shift
        rlea = lb + levn/div*(shift + 1)
        return self._nodes[level].intersect(IntervalTree.from_tuples([(llea, rlea)]))

    def parent(self, nid, check=True, level=None):
        if nid > 2:
            if check:
                if not level:
                    level = self.level(nid)
                if not self._nodes[level - 1].overlaps((nid - 1) / 2):
                    return None
            return (nid - 1) / 2
        else:
            return 0

    def children(self, nid, upto=None):
        level = self.level(nid)
        if not level:
            return NBTree(0)
        if not upto:
            upto = self._depth
        children = dict()
        for lev in range(level + 1, upto + 1):
            children[lev] = self.leaves(nid, lev)
        return NBTree(self._depth, nodes=children)

    def remove_node(self, nid, level=None):
        if not level:
            level = self.level(nid)
        self._nodes[level].chop(nid, nid + 1)

    def remove_subtree(self, nid):
        for lev, tree in self.children(nid).get_nodes():
            self._nodes[lev].subtract(tree)
        self.remove_node(nid)

    def path_to_node(self, nid):
        path = [nid]
        curnid = nid
        for lev in range(self.level(nid), 0, -1):
            curnid = self.parent(curnid, level=lev)
            path.append(curnid)
        return path

    def path_to_node_iter(self, nid):
        level = self.level(nid)
        curnid = nid
        for lev in range(level, 0, -1):
            if lev == level:
                yield nid
            else:
                curnid = self.parent(curnid, level=lev)
                yield curnid


