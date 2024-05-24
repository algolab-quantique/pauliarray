from typing import Tuple
from numpy.typing import ArrayLike


def is_broadcastable(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
    """
    Checks if two shapes are broadcastable.

    Args:
        shape1 (tuple): Shape of the first array.
        shape2 (tuple): Shape of the second array.

    Returns:
        bool: True if the shapes are broadcastable, False otherwise.
    """
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def broadcast_shape(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Calculates the broadcasted shape given two input shapes.

    Args:
        shape1 (tuple): Shape of the first array.
        shape2 (tuple): Shape of the second array.

    Returns:
        tuple: The broadcast shape.
    """
    ndim = max(len(shape1), len(shape2))
    if len(shape1) < ndim:
        shape1 += (1,) * (ndim - len(shape1))
    if len(shape2) < ndim:
        shape2 += (1,) * (ndim - len(shape2))
    shape = tuple()
    for a, b in zip(shape1, shape2):
        shape += (max(a, b),)

    return shape


def broadcasted_index(shape: Tuple[int, ...], idx: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Modifies an index to fit the broadcasted shape.

    Args:
        shape (tuple): The broadcast shape.
        idx (tuple): The original index.

    Returns:
        tuple: The modified index.
    """
    bc_idx = tuple([i if shape[dim] > 1 else 0 for dim, i in enumerate(idx)])

    return bc_idx


def is_concatenatable(arrays: ArrayLike, axis=0) -> bool:
    """
    Checks if a list of arrays can be concatenated along a given axis.

    Args:
        arrays (list): List of NumPy arrays.
        axis (int, optional): The axis along which concatenation is done. Defaults to 0.

    Returns:
        bool: True if concatenation is possible, False otherwise.
    """
    shapes = [array.shape for array in arrays]

    if not all_equal([len(shape) for shape in shapes]):
        return False
    if not axis < len(shapes[0]):
        return False
    for i in range(len(shapes[0])):
        if i == axis:
            continue
        if not all_equal([shape[i] for shape in shapes]):
            return False
    return True


def new_concatenate_shape(arrays: ArrayLike, axis=0) -> Tuple[int, ...]:
    """
    Calculates the shape after concatenating a list of arrays along a given axis.

    Args:
        arrays (list): List of NumPy arrays.
        axis (int, optional): The axis along which concatenation is done. Defaults to 0.

    Returns:
        tuple: The shape after concatenation.
    """
    shapes = [array.shape for array in arrays]

    new_shape = shapes[0][:axis] + (sum([shape[axis] for shape in shapes]),) + shapes[0][axis + 1 :]

    return new_shape


def is_stackatable(arrays: ArrayLike, axis=0) -> bool:
    """
    Checks if a list of arrays can be stacked along a given axis.

    Args:
        arrays (ArrayLike): List of NumPy arrays.
        axis (int, optional): The axis along which stacking is done. Defaults to 0.

    Returns:
        bool: True if stacking is possible, False otherwise.
    """
    shapes = [array.shape for array in arrays]

    if not all_equal([len(shape) for shape in shapes]):
        return False
    if not axis < len(shapes[0]):
        return False
    for i in range(len(shapes[0])):
        if not all_equal([shape[i] for shape in shapes]):
            return False
    return True


def all_equal(lst: ArrayLike) -> bool:
    """
    Checks if all elements in a list are equal.

    Args:
        lst (list): List of elements to check.

    Returns:
        bool: True if all elements are equal, False otherwise.
    """
    return lst[:-1] == lst[1:]
