from math import isqrt, log
from bisect import bisect_right
from itertools import takewhile
from collections import Counter


class MathAlgorithms:
  def __init__(self):
    self.factor_list = None
    self.primes_list = None
    self.num_prime_factors = None
    self.sum_prime_factors = None
    self.catalan_numbers = None

  def block_sieve_odd(self, limit):
    limit += 10
    end_sqrt, end_limit = isqrt(limit) + 1, (limit - 1) // 2
    sieve_and_block, primes, smaller_primes = [True] * (end_sqrt + 1), [2], []
    app, smaller_app = primes.append, smaller_primes.append
    for prime in range(3, end_sqrt, 2):
      if sieve_and_block[prime]:
        smaller_app([prime, (prime * prime - 1)//2])
        for j in range(prime * prime, end_sqrt + 1, prime * 2):
          sieve_and_block[j] = False
    for low in range(0, end_limit, end_sqrt):
      for i in range(end_sqrt):
        sieve_and_block[i] = True
      for i, [p, idx] in enumerate(smaller_primes):
        for idx in range(idx, end_sqrt, p):
          sieve_and_block[idx] = False
        smaller_primes[i][1] = idx - end_sqrt + (0 if idx >= end_sqrt else p)
      if low == 0:
        sieve_and_block[0] = False
      for i in range(min(end_sqrt, (end_limit + 1) - low)):
        if sieve_and_block[i]:
          app((low + i) * 2 + 1)
    self.primes_list = primes
    while self.primes_list[-1] > limit-10:
      self.primes_list.pop()

  def prime_factorize_n(self, n: int) -> None:  # using this for testing
    """A basic prime factorization of n function. without primes its just O(sqrt(n))

    Complexity: Time: O(sqrt(n)/ln(sqrt(n))), Space: O(log n)
    Variants: number and sum of prime factors, of diff prime factors, of divisors, and euler phi
    """
    limit, prime_factors = isqrt(n) + 1, []
    for prime in takewhile(lambda x: x < limit, self.primes_list):
      if n % prime == 0:
        while n % prime == 0:
          n //= prime
          prime_factors.append(prime)
    if n > 1:  # n is prime or last factor of n is prime
      prime_factors.append(n)
    self.factor_list = Counter(prime_factors)

  def factorial_prime_factors(self, limit: int) -> None:
    """This uses similar idea to sieve but avoids divisions. Complexity function 3."""
    end_point = bisect_right(self.primes_list, limit)
    prime_factors = [0] * end_point
    for i in range(end_point):
      prime, prime_amount = self.primes_list[i], 0
      exponent_limit = int(log(limit, prime)) + 2
      for exponent in range(1, exponent_limit):
        prime_amount += (limit // prime ** exponent)
      prime_factors[i] = prime_amount
    self.num_prime_factors = prime_factors

  def catalan_via_prime_factors_faster(self, n: int, k: int, mod_m: int) -> int:
    """Compute the nth Catalan number mod_n via prime factor reduction of C(2n, n)/(n+1).
    Notes: The function "num_and_sum_of_prime_factors" needs to be modified for computing number
    of each prime factor in all the numbers between 1-2n or 1 to n.

    Complexity per call: Time: O(max(n lnln(sqrt(n)), n)), Space: O(n/ln(n)).
    """
    top, bottom, ans = n, k, 1  # n = 2n and k = n in C(n, k) = (2n, n)
    self.block_sieve_odd(n)  # or use any prime sieve this one is fastest in python
    self.factorial_prime_factors(top)
    top_factors = [amt for amt in self.num_prime_factors]
    self.factorial_prime_factors(bottom)  # will handle n!n! in one go stored in num
    self.prime_factorize_n(k + 1)  # factorizing here is faster than doing n! and (n+1)!
    for prime_ind, exponent in enumerate(self.num_prime_factors):
      top_factors[prime_ind] -= (2 * exponent)
    for prime, exponent in self.factor_list.items():
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
    self.block_sieve_odd(n)
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
