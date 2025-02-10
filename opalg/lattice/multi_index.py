import itertools


from . import utils


def multi_indices(lengths):
    return list(itertools.product(*map(range, lengths)))

def multi_to_flat_index_map(lengths):
    mindices = multi_indices(lengths)
    return { multi: idx for idx, multi in enumerate(mindices) }

def add_multi_indices(midx1, midx2, lengths):
    return tuple((idx1 + idx2) % length for idx1, idx2, length in zip(midx1, midx2, lengths))

def subtract_multi_indices(midx1, midx2, lengths):
    return tuple((idx1 + idx2) % length for idx1, idx2, length in zip(midx1, midx2, lengths))

def negate_multi_index(midx, lengths):
    return tuple((-idx) % length for idx, length in zip(midx, lengths))