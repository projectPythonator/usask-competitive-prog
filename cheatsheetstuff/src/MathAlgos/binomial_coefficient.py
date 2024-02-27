from functools import lru_cache


class MathAlgorithms:
  def __init__(self):
    self.mod_p = None
    self.binomial = {}
    self.fact = []
    self.inv_fact = []

  def c_n_k(self, n: int, k: int, inverse: bool) -> int:
    """Computes C(n, k) % p. From competitive programming 4.

    Complexity per call: Time: v1 = O(log n), v2 = O(1), Space: O(1).
    v1 is uncommented, v2 is the commented out line, and must be precomputed see below.
    """
    if n <= k or k == 0:  # base case: could flip them to be n, k = k, n but better to just return 0
      return 0 if n < k else 1
    if inverse:
      return 0 if n < k else (self.fact[n] * self.inv_fact[k] * self.inv_fact[n-k]) % self.mod_p
    n_fact, k_fact, n_k_fact, p = self.fact[n], self.fact[k], self.fact[n - k], self.mod_p
    return (n_fact * pow(k_fact, p - 2, p) * pow(n_k_fact, p - 2, p)) % p

  def binomial_coefficient_n_mod_p_prep(self, max_n: int, mod_p: int, inverse: bool):
    """Does preprocessing for binomial coefficients. From competitive programming 4.

    Complexity per call: Time: v1 O(n), v2 = O(n), Space: O(n).
    Optimization and notes: v2 -> uncomment lines for C(n, k) % p in O(1) time, see above.
    """
    factorial_mod_p = [1] * max_n
    for i in range(1, max_n):
      factorial_mod_p[i] = (factorial_mod_p[i - 1] * i) % mod_p
    self.mod_p, self.fact = mod_p, factorial_mod_p
    if inverse:
      inverse_factorial_mod_p = [0] * max_n
      inverse_factorial_mod_p[-1] = pow(factorial_mod_p[-1], mod_p-2, mod_p)
      for i in range(max_n-2, -1, -1):
        inverse_factorial_mod_p[i] = (inverse_factorial_mod_p[i+1] * (i+1)) % mod_p
      self.inv_fact = inverse_factorial_mod_p

  @lru_cache(maxsize=None)
  def binomial_coefficient_dp_with_cache(self, n: int, k: int) -> int:
    """Uses the recurrence to calculate binomial coefficient. Cached for memoization.

    Complexity per call: Time: O(n*k), Space: O(n*k).
    """
    if n == k or 0 == k:
      return 1
    take_case = self.binomial_coefficient_dp_with_cache(n-1, k)
    skip_case = self.binomial_coefficient_dp_with_cache(n-1, k-1)
    return take_case + skip_case
