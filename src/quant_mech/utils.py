import functools

import numpy as np


def format_size(size_in_bytes):
    """
    Convert a size in bytes to a human-readable string with units B, KB, MB, or GB.

    Args:
        size_in_bytes (int): Size in bytes. Must be non-negative.

    Returns:
        str: Formatted size with appropriate unit.

    Raises:
        ValueError: If size_in_bytes is negative.

    Examples:
        >>> format_size(123)
        '123.00 B'
        >>> format_size(1024)
        '1.00 KB'
        >>> format_size(1048576)
        '1.00 MB'
        >>> format_size(1073741824)
        '1.00 GB'
        >>> format_size(2**40)
        '1024.00 GB'
    """
    if size_in_bytes < 0:
        raise ValueError("Size must be non-negative")

    units = ["B", "KB", "MB", "GB"]
    unit_index = 0

    # Keep dividing by 1024 to find the appropriate unit, but stop at GB
    while size_in_bytes >= 1024 and unit_index < len(units) - 1:
        size_in_bytes /= 1024
        unit_index += 1

    # Format the number to two decimal places
    return f"{size_in_bytes:.2f} {units[unit_index]}"


def print_multi_particle_states_info(multi_particle_states):

    num_particles = len(multi_particle_states[-1])
    print(f"{len(multi_particle_states)} {num_particles}-particle states")
    print(f"{format_size(len(multi_particle_states) * 16)} per state (complex128)")
    print(
        f"{format_size((len(multi_particle_states) ** 2) * 16)} for dense representation of an operator (complex128)"
    )


def create_index_map(elements):
    return {element: idx for idx, element in enumerate(elements)}


def order(lhs, rhs):
    site_index1, spin1 = lhs
    site_index2, spin2 = rhs

    if site_index1 < site_index2:
        return True
    elif site_index1 == site_index2:
        return spin1 > spin2
    return False


key_func = functools.cmp_to_key(order)


def sorted_multi_particle_state(multi_particle_state):
    """return sorted multi_particle_state as tuple and parity of swap count
    partiy = 1 if sorted list is even permutation of original list.
    parity = -1 if it is odd permutation.
    parity = 0 if it cannot be determined. (there exists a pair of identical element.)
    """
    lst = list(multi_particle_state)
    n_particles = len(lst)
    parity = 1
    for i in range(n_particles):
        swapped = False
        for j in range(0, n_particles - i - 1):
            if lst[j] == lst[j + 1]:
                parity = 0
                continue
            if not order(lst[j], lst[j + 1]):
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                swapped = True
                parity *= -1
        if not swapped:
            break
    return tuple(lst), parity
