import numpy as np
import numba as nb


@nb.njit
def bit_length(n: int):
    return int(np.log2(n)) + 1


@nb.njit
def ffs(x: int) -> int:
    """Returns the index, counting from 0, of the least significant set bit in x

    Args:
        x (int): Integer representation of a bitstring

    Returns:
        int: The index of the first set bit in the binary representation of x
    """
    return bit_length(x & -x) - 1


@nb.njit
def igray(n: int, parity: int):
    """Function for iterating a grey code and indicating the index of the changed bit

    Args:
        n (int): Grey code to be iterated
        parity (int): The parity of the input grey code (0 or 1)

    Returns:
        tuple: a tuple containing:
            - The next grey code after n
            - The index of the changed bit
    """
    if parity == 0:
        return n ^ 1, -1
    elif parity == 1:
        y = n & (~n + 1)
        return n ^ (y << 1), -2 - ffs(n)


@nb.njit
def reverse(M: np.ndarray) -> np.ndarray:
    # M is a matrix 1 x 2
    return np.array([M[0][1], M[0][0]])


@nb.njit
def subperms1(M: np.ndarray) -> np.ndarray:
    """Permanents of submatrices from the Laplace expansion of the permanent
    computed according to Lemma 2 of https://arxiv.org/abs/1706.01260
    M should have n_columns == n_rows + 1

    Args:
        M (np.ndarray): Complex rectangular matrix with (number of columns) =
        (number of rows) + 1

    Returns:
        np.ndarray: Permanents of submatrices as per Lemma 2 of https://arxiv.org/abs/1706.01260
    """
    n_cols = M.shape[1]
    n_rows = M.shape[0]

    assert n_cols == n_rows + 1

    # For 2 photons it's easy
    if n_rows == 1:
        # return np.fliplr(M)[0]
        return reverse(M)

    # Hard code it for 3 photons too
    elif n_rows == 2:
        ap = M[0, 0] + M[1, 0]
        am = M[0, 0] - M[1, 0]

        bp = M[0, 1] + M[1, 1]
        bm = M[0, 1] - M[1, 1]

        cp = M[0, 2] + M[1, 2]
        cm = M[0, 2] - M[1, 2]

        fb_1 = np.array([cp * bp, ap * cp, ap * bp])
        fb_2 = np.array([cm * bm, am * cm, am * bm])

        return 0.5 * (fb_1 - fb_2)

    # The first step with (processed, i.e. 1s and -1s instead of 0s and 1s)
    # Gray code all 1s
    v = np.sum(M, axis=0)
    f = np.cumprod(v[:-1])
    B = np.cumprod(v[1:][::-1])[::-1]

    f_b = np.zeros(n_cols, dtype=np.complex128)
    partial_sums = np.zeros(n_cols, np.complex128)
    f_b[1:-1] = np.multiply(f[:-1], B[1:])

    f_b[-1] = f[-1]
    f_b[0] = B[0]
    partial_sums += f_b

    # Initialise the Gray code for the loop
    gc = 1
    parity = 1
    ind = -1

    # The number of bits in the Gray code is the number of rows,
    # but we don't want to change the most significant bit so we keep iterating
    # until the index is that of the most significant bit
    while -ind < n_rows:
        # add if the bit at the index is 0, subtract if it is 1
        coeff = 2.0 * (-1) ** bool(gc & (1 << -ind - 1))

        v += coeff * M[ind]

        # Calculate {f_l}, {B_l} and subsequently {f_l*B_l}, each of these
        # should take k-2 multiplications.
        # f and B are the forwards and backwards partial product vectors with
        # f_0 = B_(k-1) = 1
        f = np.cumprod(v[:-1])
        B = np.cumprod(v[1:][::-1])[::-1]

        f_b[1:-1] = np.multiply(f[:-1], B[1:])
        f_b[-1] = f[-1]
        f_b[0] = B[0]

        partial_sums += f_b * (-1) ** parity

        gc, ind = igray(gc, parity)

        parity = (parity + 1) % 2

    return partial_sums / 2.0 ** (n_cols - 2)


@nb.njit
def ggc(a: np.ndarray):
    """Function for computing all the iterations of Generalised Gray Code for a given array

    Args:
        a (np.ndarray): Array of which we compute Generalised Gray code iterations

    Returns:
        output (np.ndarray): A matrix (sum(a) by 3) containing:
        - New value of changed index
        - Index changed
        - Parity of change (+1 if increased and -1 if decreased)
        for each iteration of the Generalised Gray Code
    """
    l = len(a)
    t = [0] * l
    d = [1] * l
    x = 1

    # Calculate the number of Generalised Gray Code iterations we will have
    for j in range(l):
        x *= a[j] + 1

    output = np.zeros(3 * (x - 1)).reshape(x - 1, 3)

    for j in range(x - 1):
        i = 0

        # Increase index until reaching smallest index in correct range when iterating
        # Then iterate that index
        while not 0 <= t[i] + d[i] <= a[i]:
            d[i] *= -1
            i += 1

        t[i] += d[i]

        output[j, 0] = t[i]
        output[j, 1] = i
        output[j, 2] = d[i]

    return output


@nb.njit
def subperms2(A: np.ndarray, s_values: np.ndarray) -> np.ndarray:
    """Permanents of submatrices from the Laplace expansion of the permanent
    computed according to Lemma 1 of https://arxiv.org/abs/2005.04214

    Args:
        A (np.ndarray): Complex matrix
        s_values (np.ndarray): Array of multiplicies of rows of A in the full matrix that we are computing subpermanents of

    Returns:
        np.ndarray: Permanents of submatrices as per Lemma 1 of https://arxiv.org/abs/2005.04214
    """
    n_cols = A.shape[1]
    assert n_cols == sum(s_values) + 1

    v = np.zeros(n_cols, dtype=np.complex128)
    f_b = np.zeros(n_cols, dtype=np.complex128)
    s = np.full(n_cols, 1, dtype=np.complex128)
    perm = np.zeros(n_cols, dtype=np.complex128)

    # Store all the relevant Generalised Gray Code values
    x = ggc(s_values)

    for i in range(x.shape[0]):
        v += int(x[i][2]) * A[int(x[i][1])]

        # Calculate {f_l}, {B_l} and subsequently {f_l*B_l}, each of these
        # should take k-2 multiplications.
        # f and B are the forwards and backwards partial product vectors with
        # f_0 = B_(k-1) = 1
        f = np.cumprod(v[:-1])
        B = np.cumprod(v[1:][::-1])[::-1]

        f_b[1:-1] = np.multiply(f[:-1], B[1:])
        f_b[-1] = f[-1]
        f_b[0] = B[0]

        k = int(x[i][0])

        # Mulitply by the relevant prefactor depending on if the altered Generalised Gray Code index has increased or decreased
        if int(x[i][2]) == 1:
            s = np.multiply(s, np.full(n_cols, (-1) * (s_values[int(x[i][1])] - k + 1) / k))

        else:
            s = np.multiply(s, np.full(n_cols, (-1) * (k + 1) / (s_values[int(x[i][1])] - k)))

        perm = np.add(perm, np.multiply(s, f_b))

    # Multiply the calculated values by parity factor to obtain subpermanents
    perm = np.multiply(perm, np.full(n_cols, (-1) ** (sum(s_values))))

    return perm
