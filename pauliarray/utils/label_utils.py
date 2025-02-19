import numpy as np


def table_1d(labels) -> str:

    return "\n".join(labels)


def table_2d(labels) -> str:

    row_strs = []
    for i in range(labels.shape[0]):
        row_strs.append("  ".join(labels[i, :]))

    return "\n".join(row_strs)


def table_nd(labels) -> str:

    slice_strs = []
    for idx in np.ndindex(labels.shape[:-2]):
        slice_str = "Slice (" + ",".join([str(i) for i in idx]) + ",:,:)\n"
        slice_str += table_2d(labels[idx])
        slice_strs.append(slice_str)

    return "\n".join(slice_strs)


def weighted_table_1d(labels, weights) -> str:

    pauli_str_len = len(max(labels, key=len))

    row_strs = []
    for label, weight in zip(labels, weights):
        row_strs.append(f"({weight.real:+7.4f} {weight.imag:+7.4f}j) {label:{pauli_str_len}s}")

    return "\n".join(row_strs)


def weighted_table_2d(labels, weights) -> str:

    pauli_str_len = len(max(labels, key=len))

    row_strs = []
    for i in range(labels.shape[0]):
        col_strs = []
        for label, weight in zip(labels[i, :], weights[i, :]):
            col_strs.append(f"({weight.real:+7.4f} {weight.imag:+7.4f}j) {label:{pauli_str_len}s}")
        row_strs.append("  ".join(col_strs))

    return "\n".join(row_strs)


def weighted_table_nd(labels, weights) -> str:

    slice_strs = []
    for idx in np.ndindex(labels.shape[:-2]):
        slice_str = "Slice (" + ",".join([str(i) for i in idx]) + ",:,:)\n"
        slice_str += weighted_table_2d(labels[idx], weights[idx])
        slice_strs.append(slice_str)

    return "\n".join(slice_strs)
