from typing import Tuple, List
from math import gcd, prod


class MathAlgorithms:
  def extended_euclid_recursive(self, a: int, b: int) -> Tuple[int, int, int]:
    """Solves coefficients of Bezout identity: ax + by = gcd(a, b), recursively

    Complexity per call: Time: O(log n), Space: O(log n) at the deepest call.
    """
    if 0 == b:
      return 1, 0, a
    x, y, d = self.extended_euclid_recursive(b, a % b)
    return y, x-y*(a//b), d

  def extended_euclid_iterative(self, a: int, b: int) -> Tuple[int, int, int]:
    """Solves coefficients of Bezout identity: ax + by = gcd(a, b), iteratively.

    Complexity per call: Time: O(log n) about twice as fast in python vs above, Space: O(1)
    Optimizations and notes:
      divmod and abs are used to help deal with big numbers, remove if < 2**64 for speedup.
    """
    last_remainder, remainder = abs(a), abs(b)
    x, y, last_x, last_y = 0, 1, 1, 0
    while remainder:
      last_remainder, (quotient, remainder) = remainder, divmod(last_remainder, remainder)
      x, last_x = last_x - quotient * x, x
      y, last_y = last_y - quotient * y, y
    return -last_x if a < 0 else last_x, -last_y if b < 0 else last_y, last_remainder

  def safe_modulo(self, a: int, n: int) -> int:
    """Existence is much for c++ which doesn't always handle % operator nicely.
    use ((a % n) + n) % n for getting proper mod of a potential negative value
    use (a + b) % --> ((a % n) + (b % n)) % n for operations sub out + for * and -
    """
    return ((a % n) + n) % n

  def modular_linear_equation_solver(self, a: int, b: int, n: int) -> List[int]:  # TODO RETEST
    """Solves gives the solution x in ax = b(mod n).

    Complexity per call: Time: O(log n), Space: O(d)
    """
    x, y, d = self.extended_euclid_iterative(a, n)
    if 0 == b % d:
      x = (x * (b//d)) % n
      return [(x + i*(n//d)) % n for i in range(d)]
    return []

  def linear_diophantine_1(self, a: int, b: int, c: int) -> Tuple[int, int]:  # TODO RETEST
    """Solves for x, y in ax + by = c. From stanford icpc 2013-14

    Complexity per call: Time: O(log n), Space: O(1).
    Notes: order matters? 25x + 18y = 839 != 18x + 25y = 839
    """
    d = gcd(a, b)
    if c % d == 0:
      x = c//d * self.mod_inverse(a//d, b//d)
      return x, (c - a * x) // b
    return -1, -1

  def linear_diophantine_2(self, a: int, b: int, c: int) -> Tuple[int, int]:  # TODO RETEST
    """Solves for x0, y0 in x = x0 + (b/d)n, y = y0 - (a/d)n.
    derived from ax + by = c, d = gcd(a, b), and d|c.
    Can further derive into: n = x0 (d/b), and n = y0 (d/a).

    Complexity per call: Time: O(log n), Space: O(1).
    Optimizations and notes:
      unlike above this function order doesn't matter if a != b
      for a speedup call math.gcd(a, b) at start and return accordingly on two lines
    """
    x, y, d = self.extended_euclid_iterative(a, b)
    return (-1, -1) if c % d != 0 else (x * (c // d), y * (c // d))

  def mod_inverse(self, b: int, m: int) -> None | int:
    """Solves b^(-1) (mod m).

    Complexity per call: Time: O(log n), Space: O(1)
    """
    x, y, d = self.extended_euclid_iterative(b, m)
    return None if d != 1 else x % m  # -1 instead of None if we intend to go on with the prog

  def chinese_remainder_theorem_1(self, remainders: List[int], modulos: List[int]) -> int:
    """Steven's CRT version to solve x in x = r[0] (mod m[0]) ... x = r[n-1] (mod m[n-1]).
    # TODO RETEST
    Complexity per call: Time: O(n log n), Space: O(1)? O(mt) bit size:
    Optimizations:
      prod is used from math since 3.8,
      we use mod mt in the forloop since x might get pretty big.
    """
    mt, x = prod(modulos), 0
    for i, modulo in enumerate(modulos):
      p = mt // modulo
      x = (x + (remainders[i] * self.mod_inverse(p, modulo) * p)) % mt
    return x

  def chinese_remainder_theorem_helper(self, mod1: int, rem1: int,
                                       mod2: int, rem2: int) -> Tuple[int, int]:
    """Chinese remainder theorem (special case): find z such that z % m1 = r1, z % m2 = r2.
    Here, z is unique modulo M = lcm(m1, m2). Return (z, M).  On failure, M = -1.
    from: stanford icpc 2016
    # TODO RETEST
    Complexity per call: Time: O(log n), Space: O(1)
    """
    s, t, d = self.extended_euclid_iterative(mod1, mod2)
    if rem1 % d != rem2 % d:
      mod3, s_rem_mod, t_rem_mod = mod1*mod2, s*rem2*mod1, t*rem1*mod2
      return ((s_rem_mod + t_rem_mod) % mod3) // d, mod3 // d
    return 0, -1

  def chinese_remainder_theorem_2(self, remainders: List[int],
                                  modulos: List[int]) -> Tuple[int, int]:
    """Chinese remainder theorem: find z such that z % m[i] = r[i] for all i.  Note that the
    solution is unique modulo M = lcm_i (m[i]).  Return (z, M). On failure, M = -1. Note that
    we do not require the r[i]'s to be relatively prime.
    from: stanford icpc 2016
    # TODO RETEST
    Complexity per call: Time: O(n log n), Space: O(1)? O(mt) bit size
    """
    z_m = remainders[0], modulos[0]
    for i, modulo in enumerate(modulos[1:], 1):
      z_m = self.chinese_remainder_theorem_helper(z_m[1], z_m[0], modulo, remainders[i])
      if -1 == z_m[1]:
        break
    return z_m