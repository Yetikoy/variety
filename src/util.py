import functools
import operator
import itertools
from collections import defaultdict
import numpy as np

flatten = itertools.chain.from_iterable

def group_by(items, key=lambda x:x):
    if isinstance(items, dict):
        items = items.items()
    result = defaultdict(list)
    for k,v in items:
        result[key(v)].append(k)
    return result

def auc(xs, ys): return np.trapz(ys, x=xs)
