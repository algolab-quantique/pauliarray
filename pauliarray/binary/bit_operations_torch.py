import torch
from torch import Tensor


def bit_sum(bit_strings: Tensor) -> int:
    return bit_strings.to(torch.int32).sum(dim=-1).item()


def dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).to(torch.int32).sum(dim=-1)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return (a.to(torch.int32) @ b.to(torch.int32)) % 2 == 1


def add(a: Tensor, b: Tensor) -> Tensor:
    return torch.logical_xor(a, b)


def rank(bit_matrix: Tensor) -> int:
    assert bit_matrix.dim() == 2
    rs = row_space(bit_matrix)
    return rs.shape[0]


def inv(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        raise ValueError("Input matrix must be square.")

    return torch.linalg.inv(matrix.to(torch.float32)).to(torch.bool)


def strings_to_ints(bit_strings: torch.Tensor) -> torch.Tensor:
    power_of_twos = 2 ** torch.arange(bit_strings.shape[-1], device=bit_strings.device)
    return torch.sum(bit_strings * power_of_twos, dim=-1)


def row_echelon(matrix: torch.Tensor) -> torch.Tensor:
    re_bit_matrix = matrix.clone().to(torch.bool).to(matrix.device)
    n_rows, n_cols = re_bit_matrix.shape
    row_range = torch.arange(n_rows, device=matrix.device)
    h_row = 0
    k_col = 0

    while h_row < n_rows and k_col < n_cols:
        column = re_bit_matrix[h_row:, k_col]
        if torch.all(column == 0):
            k_col += 1
            continue

        i_row = h_row + torch.argmax(column.to(torch.int8))

        if i_row != h_row:
            temp = re_bit_matrix[h_row].clone()
            re_bit_matrix[h_row] = re_bit_matrix[i_row]
            re_bit_matrix[i_row] = temp

        mask = torch.logical_and(re_bit_matrix[:, k_col], row_range != h_row)

        if mask.any():
            pivot_row = re_bit_matrix[h_row].unsqueeze(0)  # Shape: (1, n_cols)
            re_bit_matrix[mask] = torch.logical_xor(re_bit_matrix[mask], pivot_row)

        h_row += 1
        k_col += 1

    return re_bit_matrix


def kernel(bit_matrix: torch.Tensor) -> torch.Tensor:
    assert bit_matrix.dim() == 2

    re_bit_matrix = bit_matrix.T

    n_rows = re_bit_matrix.shape[0]
    n_cols = re_bit_matrix.shape[1]

    identity_matrix = torch.eye(n_rows, dtype=torch.bool, device=bit_matrix.device)

    ext_bit_matrix = torch.cat([re_bit_matrix, identity_matrix], dim=1)

    row_ech_ext_bit_matrix = row_echelon(ext_bit_matrix)

    row_ech_bit_matrix = row_ech_ext_bit_matrix[:, :n_cols]
    inverse_bit_matrix = row_ech_ext_bit_matrix[:, n_cols:]

    null_rows = torch.all(~row_ech_bit_matrix, dim=1)

    result = inverse_bit_matrix[null_rows, :]
    if result.shape[0] == 0:
        result = torch.tensor([], dtype=torch.bool, device=bit_matrix.device)

    return result


def intersection_row_space(
    bit_matrix_1: torch.Tensor, bit_matrix_2: torch.Tensor
) -> torch.Tensor:

    assert bit_matrix_1.dim() == bit_matrix_2.dim() == 2
    assert bit_matrix_1.shape[1] == bit_matrix_2.shape[1]

    rs_bit_matrix_1 = row_space(bit_matrix_1)
    rs_bit_matrix_2 = row_space(bit_matrix_2)

    num_rows_1 = rs_bit_matrix_1.shape[0]

    all_rows = torch.cat((rs_bit_matrix_1, rs_bit_matrix_2), 0).to(torch.int32)

    null_row_combination = kernel(all_rows.T).to(torch.int32)

    return torch.matmul(null_row_combination[:, :num_rows_1], all_rows[:num_rows_1, :])


def row_space(bit_matrix: Tensor) -> Tensor:
    re_matrix = row_echelon(bit_matrix)
    mask = torch.any(re_matrix, dim=1)
    filtered = re_matrix[mask]
    return filtered


def rank(bit_matrix: Tensor) -> int:
    assert bit_matrix.dim() == 2
    rs = row_space(bit_matrix)
    return rs.shape[0]


def orthogonal_basis(bit_strings: torch.Tensor) -> torch.Tensor:
    return row_space(bit_strings)


def orthogonal_complement(bit_strings: torch.Tensor) -> torch.Tensor:
    return kernel(bit_strings)


def intersection(bit_strings_1: torch.Tensor, bit_strings_2: torch.Tensor) -> torch.Tensor:
    return intersection_row_space(bit_strings_1, bit_strings_2)


def is_orthogonal(bit_strings_1: torch.Tensor, bit_strings_2: torch.Tensor) -> torch.Tensor:
    assert bit_strings_1.shape[-1] == bit_strings_2.shape[-1]
    assert bit_strings_1.shape[-1] % 2 == 0

    return ~(dot(bit_strings_1, bit_strings_2))

