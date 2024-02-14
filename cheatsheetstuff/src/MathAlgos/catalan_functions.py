from math import isqrt, log
from bisect import bisect_right
from itertools import takewhile
from collections import Counter
import prime_sieves
import factorizations


class MathAlgorithms:
  def __init__(self):
    self.factor_list = None
    self.primes_list = None
    self.num_prime_factors = None
    self.sum_prime_factors = None
    self.catalan_numbers = None

    self.sieve_obj = prime_sieves.MathAlgorithms()
    self.sieve_function = self.sieve_obj.prime_sieve_super_fast_faster_maybe
    self.primes_list = []
    self.primes_set = set()

    self.factor_obj = factorizations.MathAlgorithms()
    self.factor_n_function = self.factor_obj.prime_factorize_n

  def fill_primes_list_factor_function(self, n):
    """Fills primes list using sieve function"""
    self.sieve_function(n)
    self.primes_list = self.sieve_obj.primes_list
    self.factor_obj.primes_list = self.sieve_obj.primes_list

  def factorial_prime_factors(self, limit: int) -> None:
    """This uses similar idea to sieve but avoids divisions. Complexity function 3."""
    end_point = bisect_right(self.primes_list, limit)
    prime_factors = [0] * end_point
    for i in range(end_point):
      prime, prime_amount, x = self.primes_list[i], 0, limit
      while x:
        x //= prime
        prime_amount += x
      prime_factors[i] = prime_amount
    self.num_prime_factors = prime_factors

  def catalan_via_prime_factors_faster(self, n: int, k: int, mod_m: int) -> int:
    """Compute the nth Catalan number mod_n via prime factor reduction of C(2n, n)/(n+1).
    Notes: The function "num_and_sum_of_prime_factors" needs to be modified for computing number
    of each prime factor in all the numbers between 1-2n or 1 to n.

    Complexity per call: Time: O(max(n lnln(sqrt(n)), n)), Space: O(n/ln(n)).
    """
    top, bottom, ans = n, k, 1  # n = 2n and k = n in C(n, k) = (2n, n)
    self.fill_primes_list_factor_function(n)
    self.factorial_prime_factors(top)
    top_factors = [amt for amt in self.num_prime_factors]
    self.factorial_prime_factors(bottom)  # will handle n!n! in one go stored in num
    for prime_ind, exponent in enumerate(self.num_prime_factors):
      top_factors[prime_ind] -= (2 * exponent)
    for prime, exponent in self.factor_n_function(k+1).items():  # k + 1 factor > (k+1)! factor
      top_factors[bisect_right(self.primes_list, prime) - 1] -= exponent
    for prime_ind, exponent in enumerate(top_factors):  # remember use multiplication not add
      if exponent > 0:
        ans = (ans * pow(self.primes_list[prime_ind], exponent, mod_m)) % mod_m
    return ans

  def catalan_via_prime_factors_slower(self, n: int, k: int, mod_m: int) -> int:
    """Compute the nth Catalan number mod_n via prime factor reduction of C(2n, n)/(n+1).
    Notes: The function "num_and_sum_of_prime_factors" needs to be modified for computing number
    of each prime factor in all the numbers between 1-2n or 1 to n.

    Complexity per call: Time: O(max(n lnln(sqrt(n)), n)), Space: O(n/ln(n)).
    """
    top, bottom, ans = n, k, 1  # n = 2n and k = n in C(n, k) = (2n, n)
    self.fill_primes_list_factor_function(n)
    self.factorial_prime_factors(top)
    top_factors = [amt for amt in self.num_prime_factors]
    self.factorial_prime_factors(bottom)  # will handle n!n! in one go stored in num
    for prime_ind, exponent in enumerate(self.num_prime_factors):
      top_factors[prime_ind] -= exponent
    self.factorial_prime_factors(bottom + 1)  # will handle n!n! in one go stored in num
    for prime_ind, exponent in enumerate(self.num_prime_factors):
      top_factors[prime_ind] -= exponent
    for prime_ind, exponent in enumerate(top_factors):  # remember use multiplication not add
      if exponent > 0:
        ans = (ans * pow(self.primes_list[prime_ind], exponent, mod_m)) % mod_m
    return ans

  def generate_catalan_n(self, n: int) -> None:
    """Generate catalan up to n iteratively.

    Complexity per call: Time: O(n*O(multiplication)), Space: O(n * 2^(log n)).
    """
    catalan = [0] * (n + 1)
    catalan[0] = 1
    for i in range(n - 1):
      catalan[i + 1] = catalan[i] * (4 * i + 2) // (i + 2)
    self.catalan_numbers = catalan[:-1]  # cut the last number off for inclusive reasons

  def generate_catalan_n_mod_inverse(self, n: int, p: int) -> None:
    """Generate catalan up to n iteratively cat n % p.

    Complexity per call: Time: O(n log n), Space: O(n * (2^(log n)%p)).
    Variants: use prime factors of the factorial to cancel out the primes
    """
    catalan = [0] * (n + 1)
    catalan[0] = 1
    for i in range(n - 1):
      catalan[i + 1] = (((4 * i + 2) % p) * (catalan[i] % p) * pow(i + 2, p - 2, p)) % p
    self.catalan_numbers = catalan[:-1]  # cut the last number off for inclusive reasons

#
# def test_1():
#   limit = 10**7
#   mod_m = 10**9+7
#   obj = MathAlgorithms()
#   obj.generate_catalan_n_mod_inverse(limit, mod_m)
#
#
# def test_2():
#   limit = 10**7
#   mod_m = 10**9+7
#   obj = MathAlgorithms()
#   obj.catalan_via_prime_factors_faster(2*limit, limit, mod_m)
#
#
# import cProfile
# cProfile.run("test_2()", sort='tottime')
# cProfile.run("test_1()", sort='tottime')