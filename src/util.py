import functools
import operator
import itertools

flatten = itertools.chain.from_iterable

def group_by(items, key=lambda x:x):
    if isinstance(items, dict):
        items = items.items()
    result = defaultdict(list)
    for k,v in items:
        result[key(v)].append(k)
    return result

