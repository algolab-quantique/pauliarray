import numpy as np
from numpy.typing import NDArray

INTSIZE_TO_UINTTYPE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}

"""
A bit_string is a 1D array of bool
A int_string is a 1D array of integers which bits are encoding the bit_string
A void is a variable encoding a certain number of bits
"""


def bit_strings_to_int_strings(bit_strings: NDArray[np.uint8]) -> NDArray[np.uint]:

    int_strings = np.packbits(bit_strings, axis=-1, bitorder="little")

    return int_strings


def int_strings_to_bit_strings(int_strings: NDArray[np.uint], num_qubits: int) -> NDArray[np.uint8]:

    bit_strings = np.unpackbits(int_strings, axis=-1, bitorder="little")[..., :num_qubits]

    return bit_strings.astype(np.bool)


def int_strings_to_voids(int_strings: NDArray[np.uint]) -> NDArray:
    """
    Converts int strings into voids.

    Args:
        int_strings (NDArray[np.uint]): _description_

    Returns:
        NDArray: _description_
    """

    pad_int_strings = pad_int_strings_to_commensurate_itemsize(
        int_strings, int(2 ** np.ceil(np.log2(int_strings.dtype.itemsize)))
    )

    void_type_size = pad_int_strings.dtype.itemsize * pad_int_strings.shape[-1]
    voids = np.squeeze(np.ascontiguousarray(pad_int_strings).view(np.dtype((np.void, void_type_size))), axis=-1)

    return voids


def voids_to_int_strings(voids: NDArray, int_size: int = None) -> NDArray[np.uint64]:

    if int_size is None:
        if voids.dtype.itemsize in [1, 2, 4, 8]:
            int_size = voids.dtype.itemsize
        elif voids.dtype.itemsize % 8 == 0:
            int_size = 8
        else:
            raise RuntimeError()

    int_dtype = INTSIZE_TO_UINTTYPE[int_size]

    int_strings = np.expand_dims(voids, axis=-1).view(int_dtype)

    return int_strings


def bit_strings_to_voids(bit_strings: NDArray[np.uint8]) -> NDArray:

    int_strings = bit_strings_to_int_strings(bit_strings)
    voids = int_strings_to_voids(int_strings)

    return voids


def voids_to_bit_strings(voids: NDArray, num_qubits: int) -> NDArray[np.uint8]:

    int_strings = voids_to_int_strings(voids, int_size=1)
    bit_strings = int_strings_to_bit_strings(int_strings, num_qubits)

    return bit_strings


def int_strings_bitcount(int_strings: NDArray[np.int_]):

    bitcounts = np.bitwise_count(int_strings).astype(np.int64)

    return np.sum(bitcounts, axis=-1)


def bitwise_count(voids: NDArray) -> NDArray[np.int64]:

    int_strings = voids_to_int_strings(voids)
    return int_strings_bitcount(int_strings)


def bitwise_not(voids: NDArray) -> NDArray:

    int_strings = voids_to_int_strings(voids)
    new_int_strings = np.invert(int_strings)
    new_voids = int_strings_to_voids(new_int_strings)

    return new_voids


def paded_bitwise_not(voids: NDArray, num_qubits: int) -> NDArray:

    int_strings = voids_to_int_strings(voids)
    new_int_strings = np.invert(int_strings)

    int_size = int_strings.dtype.itemsize
    last_num_qubits = num_qubits % (8 * int_size)

    if last_num_qubits > 0:
        new_int_strings[..., -1] = np.bitwise_and(new_int_strings[..., -1], 2**last_num_qubits - 1)

    new_voids = int_strings_to_voids(new_int_strings)

    return new_voids


def bitwise_dot(
    voids_1: NDArray,
    voids_2: NDArray,
) -> NDArray[np.int64]:

    assert voids_1.dtype.itemsize == voids_2.dtype.itemsize

    int_strings_1 = voids_to_int_strings(voids_1)
    int_strings_2 = voids_to_int_strings(voids_2)

    return int_strings_bitcount(np.bitwise_and(int_strings_1, int_strings_2))


def bitwise_and(
    voids_1: NDArray,
    voids_2: NDArray,
) -> NDArray:

    assert voids_1.dtype.itemsize == voids_2.dtype.itemsize

    int_strings_1 = voids_to_int_strings(voids_1)
    int_strings_2 = voids_to_int_strings(voids_2)

    int_strings_3 = np.bitwise_and(int_strings_1, int_strings_2)
    voids_3 = int_strings_to_voids(int_strings_3)

    return voids_3


def bitwise_xor(
    voids_1: NDArray,
    voids_2: NDArray,
) -> NDArray:

    assert voids_1.dtype.itemsize == voids_2.dtype.itemsize

    int_strings_1 = voids_to_int_strings(voids_1)
    int_strings_2 = voids_to_int_strings(voids_2)

    int_strings_3 = np.bitwise_xor(int_strings_1, int_strings_2)
    voids_3 = int_strings_to_voids(int_strings_3)

    return voids_3


def bitwise_or(
    voids_1: NDArray,
    voids_2: NDArray,
) -> NDArray:

    assert voids_1.dtype.itemsize == voids_2.dtype.itemsize

    int_strings_1 = voids_to_int_strings(voids_1)
    int_strings_2 = voids_to_int_strings(voids_2)

    int_strings_3 = np.bitwise_or(int_strings_1, int_strings_2)
    voids_3 = int_strings_to_voids(int_strings_3)

    return voids_3


def stich_voids(
    voids_1: NDArray,
    voids_2: NDArray,
) -> NDArray:

    assert voids_1.dtype.itemsize == voids_2.dtype.itemsize

    void_type_size = voids_1.dtype.itemsize + voids_2.dtype.itemsize
    new_voids = np.squeeze(
        np.ascontiguousarray(np.stack((voids_1, voids_2), axis=-1)).view(np.dtype((np.void, void_type_size))), axis=-1
    )

    return new_voids


def split_voids(
    voids: NDArray,
) -> NDArray:

    assert voids.dtype.itemsize % 2 == 0

    void_type_size = voids.dtype.itemsize // 2
    new_voids = np.expand_dims(voids, axis=-1).view(np.dtype((np.void, void_type_size)))

    voids_1 = new_voids[..., 0]
    voids_2 = new_voids[..., 1]

    return voids_1, voids_2


def pad_int_strings_to_commensurate_itemsize(int_strings: NDArray[np.uint], new_itemsize: int) -> NDArray[np.uint]:
    """
    Pad int_strings so that the new int_strings size is a multiple of the new_itemsize. For example, a 3 uint16 (size=2) int_string and a new_itemsize=8 would pad 1 uint16.

    Args:
        int_strings (NDArray[np.uint]): _description_
        new_itemsize (int): _description_

    Returns:
        NDArray[np.uint]: _description_
    """
    assert new_itemsize in [1, 2, 4, 8]

    tot_int_strings_size = int_strings.shape[-1] * int_strings.dtype.itemsize

    new_int_size = np.lcm(int_strings.dtype.itemsize, new_itemsize)
    tot_num_new_ints = np.ceil(tot_int_strings_size / new_int_size).astype(int)

    tot_new_int_strings_size = int(2 ** np.ceil(np.log2(tot_num_new_ints * new_itemsize)))
    tot_num_ints = tot_new_int_strings_size // int_strings.dtype.itemsize

    int_to_pad = tot_num_ints - int_strings.shape[-1]

    if int_to_pad == 0:
        pad_int_strings = int_strings
    else:
        pad_width = [(0, 0)] * (int_strings.ndim - 1)
        pad_width.append((0, int_to_pad))

        pad_int_strings = np.pad(int_strings, pad_width)

    return pad_int_strings
