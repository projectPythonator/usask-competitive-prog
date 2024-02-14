from math import cos, sin, tau
from typing import List
from itertools import accumulate
from operator import mul as operator_mul


class MathAlgorithms:
  def __init__(self):
    self.fft_lengths = []
    self.fft_swap_indices = []
    self.fft_roots_of_unity = []

  def fft_prepare_swap_indices(self, a_len: int):
    """Gives us all the swap pairs needed to for an FFT call stored in fft_swap_indices.

    Complexity per call: Time: O(n), Space: O(n), S(n/2) | n == 2^i -> minimax value of i.
    """
    a_bit_len, a_bit_half = a_len.bit_length(), (a_len.bit_length()-1)//2
    swap_size = (1 << a_bit_half) - 1
    swap_size = swap_size << (a_bit_half-1) if a_bit_len & 1 else swap_size << a_bit_half
    swaps, k, ind = [None] * swap_size, 0, -1
    for i in range(1, a_len):
      bit_mask = a_len >> 1
      while k & bit_mask:
        k = k ^ bit_mask
        bit_mask = bit_mask >> 1
      k = k ^ bit_mask
      if i < k:
        swaps[ind := ind + 1] = (i, k)
    self.fft_swap_indices = swaps

  def fft_prepare_lengths_list(self, a_len: int):
    """Function for all powers 2 from 2 - a_len, inclusive. O(log n) complexity."""
    self.fft_lengths = [2**power for power in range(1, a_len.bit_length())]

  def fft_prepare_roots_helper(self, length: int, angle: float) -> List[complex]:
    """Precomputes roots of unity for a given length and angle. accumulate used here :).

    Complexity per call: Time: O(n), Space: O(n) | n == 2^i.
    """
    initial_root_of_unity = complex(1)
    multiplier = complex(cos(angle), sin(angle))
    return list(accumulate([multiplier] * length, operator_mul, initial=initial_root_of_unity))

  def fft_prepare_roots_of_unity(self, invert: bool):
    """Precomputes all roots of unity for all lengths. Stores the result for later use.

    Complexity per call: Time: O(2^((log n)+1)-1) = O(n),
              Space: O(n), S(2^((log n)+1)-1) | n == len of our data, which is 2^i.
    """
    signed_tau: float = -tau if invert else tau
    self.fft_roots_of_unity = [self.fft_prepare_roots_helper(length//2 - 1, signed_tau/length)
                               for length in self.fft_lengths]

  def fft_in_place_fast_fourier_transform(self, a_vector: List[complex], invert: bool):
    """Optimized in-place Cooley-Tukey FFT algorithm. Modifies a_vector.

    Complexity per call: Time: O(n log n), Space: O(1) | technically O(n) but we precompute.
    Optimizations: swap indices, lengths, and roots of unity all have be calculated beforehand.
      This allows us to only do those once in when doing multiplication
    """
    a_len = len(a_vector)
    for i, j in self.fft_swap_indices:
      a_vector[i], a_vector[j] = a_vector[j], a_vector[i]
    for k, length in enumerate(self.fft_lengths):  # is [2, 4, 8..., 2^(i-1), 2^i] | n == 2^i
      j_end = length // 2  # j_end is to avoid repeated divisions in the innermost loop
      for i in range(0, a_len, length):
        for j, w in enumerate(self.fft_roots_of_unity[k]):
          i_j, i_j_j_end = i + j, i + j + j_end
          u, v = a_vector[i_j], w * a_vector[i_j_j_end]
          a_vector[i_j], a_vector[i_j_j_end] = u + v, u - v
    if invert:
      a_vector[:] = [complex_number/a_len for complex_number in a_vector]

  def fft_normalize(self, a_vector: List[int], base: int) -> List[int]:
    """Normalizes polynomial a for a given base. base 10 will result in a base 10 number

    Complexity per call: Time: O(n), Space: O(1) -> in fact we often reduce overall space.
    """
    carry, end = 0, len(a_vector) - 1
    for i in range(end + 1):
      carry, a_vector[i] = divmod(a_vector[i] + carry, base)
    while 0 == a_vector[end]:
      end -= 1
    return a_vector[:end+1][::-1]

  def fft_multiply_in_place(self, polynomial_a: List[int], polynomial_b: List[int]) -> List[int]:
    """Multiply two polynomials with the option to normalize then after.

    Complexity per call: Time: O(n log n), T(4(n log n)),
              Space: O(n),  S(4n) | [n == |a| + |b|]
    Optimizations: listed fft_in_place_fast_fourier_transform.
    """
    a_len, b_len = len(polynomial_a), len(polynomial_b)
    n = 2**((a_len + b_len).bit_length())     # computes 2^(log2(a+b) + 1)
    n = n if (a_len + b_len) != n//2 else n//2  # optimization that fixes n when (a+b) % 2 == 0
    a_vector = [complex(i) for i in polynomial_a] + [complex(0)] * (n - a_len)
    b_vector = [complex(i) for i in polynomial_b] + [complex(0)] * (n - b_len)
    self.fft_prepare_swap_indices(n)    # these three calls are for optimization with
    self.fft_prepare_lengths_list(n)    # multiplying, if calling fft outside multiply
    self.fft_prepare_roots_of_unity(False)  # you will need to call for each size of array.
    self.fft_in_place_fast_fourier_transform(a_vector, False)
    self.fft_in_place_fast_fourier_transform(b_vector, False)
    a_vector = [i * j for i, j in zip(a_vector, b_vector)]
    self.fft_prepare_roots_of_unity(True)
    self.fft_in_place_fast_fourier_transform(a_vector, True)
    a_vector = [int(round(el.real)) for el in a_vector]   # optional
    return self.fft_normalize(a_vector, 10)       # optional turns into base 10 num
