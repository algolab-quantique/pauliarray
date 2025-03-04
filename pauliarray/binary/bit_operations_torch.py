import torch
from torch import Tensor


def bit_sum(bit_strings: Tensor) -> Tensor:
    return bit_strings.to(torch.int32).sum(dim=-1)


def dot(a: Tensor, b: Tensor) -> Tensor:
    return (a & b).to(torch.int32).sum(dim=-1) % 2


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return (a.to(torch.int32) @ b.to(torch.int32)) % 2 == 1


def add(a: Tensor, b: Tensor) -> Tensor:
    return torch.logical_xor(a, b)


def rank(bit_matrix: Tensor) -> int:
    assert bit_matrix.dim() == 2
    rs = row_space(bit_matrix)
    return rs.shape[0]


def inv(bit_matrix: torch.Tensor) -> torch.Tensor:
    assert bit_matrix.dim() == 2
    float_matrix = bit_matrix.float()
    inverse = torch.inverse(float_matrix)
    binary_inverse = (inverse > 0.5).int()
    return binary_inverse


def strings_to_ints(bit_strings: torch.Tensor) -> torch.Tensor:
    power_of_twos = 2 ** torch.arange(bit_strings.shape[-1], device=bit_strings.device)
    return torch.sum(bit_strings * power_of_twos, dim=-1)


def row_echelon(matrix: torch.Tensor) -> torch.Tensor:
    device = matrix.device
    re_matrix = matrix.clone().to(torch.bool)
    n_rows, n_cols = re_matrix.shape
    current_row = 0
    pivot_col = 0

    # Transform into row echelon form without sorting during elimination
    while current_row < n_rows and pivot_col < n_cols:
        rows_with_ones = torch.where(re_matrix[current_row:, pivot_col])[0]
        if rows_with_ones.numel() == 0:
            pivot_col += 1
            continue

        target_row = rows_with_ones[0] + current_row
        if target_row != current_row:
            re_matrix[[current_row, target_row]] = re_matrix[[target_row, current_row]]

        # Eliminate below the pivot
        mask = (re_matrix[:, pivot_col]) & (torch.arange(n_rows, device=device) > current_row)
        if mask.any():
            re_matrix[mask] = re_matrix[mask] != re_matrix[current_row]  # XOR for bool

        current_row += 1
        pivot_col += 1

    # Sort rows by leading 1 position after elimination
    lead_positions = torch.full((n_rows,), n_cols, device=device, dtype=torch.long)
    for i in range(n_rows):
        ones = torch.where(re_matrix[i])[0]
        if ones.numel() > 0:
            lead_positions[i] = ones[0]
    _, indices = torch.sort(lead_positions)
    return re_matrix[indices]


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

    return inverse_bit_matrix[null_rows, :]


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
    print("Row echelon form:\n", re_matrix.int())  # Convert to int for readability
    mask = torch.any(re_matrix, dim=1)
    print("Filter mask:", mask)
    filtered = re_matrix[mask]
    print("Filtered row space:\n", filtered.int())
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