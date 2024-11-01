import numpy as np


def label_table_1d(labels) -> str:

    return "\n".join(labels)


def label_table_2d(labels) -> str:

    row_strs = []
    for i in range(labels.shape[0]):
        row_strs.append("  ".join(labels[i, :]))

    return "\n".join(row_strs)


def label_table_nd(labels) -> str:

    slice_strs = []
    for idx in np.ndindex(labels.shape[:-2]):
        slice_str = "Slice (" + ",".join([str(i) for i in idx]) + ",:,:)\n"
        slice_str += label_table_2d(labels[idx])
        slice_strs.append(slice_str)

    return "\n".join(slice_strs)
