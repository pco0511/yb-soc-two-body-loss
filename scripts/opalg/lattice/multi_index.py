import itertools


from . import utils


def multi_indices(lengths):
    return list(itertools.product(*map(range, lengths)))

def multi_to_flat_index_map(lengths):
    mindices = multi_indices(lengths)
    return { multi: idx for idx, multi in enumerate(mindices) }

def add_multi_indices(midx1, midx2, mods=None):
    if mods is None:
        return tuple((idx1 + idx2) for idx1, idx2 in zip(midx1, midx2))
    return tuple((idx1 + idx2) % mod for idx1, idx2, mod in zip(midx1, midx2, mods))

def subtract_multi_indices(midx1, midx2, mods=None):
    if mods is None:
        return tuple((idx1 - idx2) for idx1, idx2 in zip(midx1, midx2))
    return tuple((idx1 - idx2) % mod for idx1, idx2, mod in zip(midx1, midx2, mods))

def negate_multi_index(midx, mods=None):
    if mods is None:
        return tuple((-idx) for idx in midx)
    return tuple((-idx) % mod for idx, mod in zip(midx, mods))

def fold(midx, mods):
    """Fold a multi-index to the range [0, mod)"""
    return tuple(idx % mod for idx, mod in zip(midx, mods))

def _min_index(lengths):
    return tuple(-((length - 1) // 2) for length in lengths)

def _max_index(lengths):
    return tuple(length // 2 for length in lengths)

def fold_bz(midx, lengths):
    """Fold a multi-index to the range [min_idx, max_idx]"""
    min_idx = _min_index(lengths)
    max_idx = _max_index(lengths)
    def fold_single(idx, min_idx, max_idx):
        if idx < min_idx:
            return max_idx - (min_idx - idx) % (max_idx - min_idx + 1)
        elif idx > max_idx:
            return min_idx + (idx - max_idx - 1) % (max_idx - min_idx + 1)
        return idx
    return tuple(fold_single(idx, m, M) for idx, m, M in zip(midx, min_idx, max_idx))

def check_constrain(midx, lengths):
    """Check if a multi-index is within the bounds defined by lengths"""
    min_idx = _min_index(lengths)
    max_idx = _max_index(lengths)
    return all(min_idx[i] <= midx[i] <= max_idx[i] for i in range(len(lengths)))