from functools import lru_cache


class MathAlgorithms:
  def __init__(self):
    self.fibonacci_list = None

  def fibonacci_n_iterative(self, n: int) -> None:
    """Classic fibonacci solver. Generates answers from 0 to n inclusive.

    Complexity per call: Time: O(n), Space: O(n).
    """
    fib_list = [0] * (n + 1)
    fib_list[1] = 1
    for i in range(2, n + 1):
      fib_list[i] = fib_list[i - 1] + fib_list[i - 2]
    self.fibonacci_list = fib_list

  @lru_cache(maxsize=None)
  def fibonacci_n_dp_cached(self, n: int) -> int:
    """Cached Dynamic programming to get the nth fibonacci. Derived from Cassini's identity.

    Complexity per call: Time: O(log n), Space: increase by O(log n).
    Optimization: can go back to normal memoization with same code but using dictionary.
    """
    if n < 3:
      return 1 if n else 0
    f1, f2 = self.fibonacci_n_dp_cached(n // 2 + 1), self.fibonacci_n_dp_cached((n - 1) // 2)
    return f1 * f1 + f2 * f2 if n & 1 else f1 * f1 - f2 * f2

  @lru_cache(maxsize=None)
  def fibonacci_n_dp_cached_faster(self, n: int) -> int:
    """Same as above but runs in ~Time*0.75 i.e. Above takes 20 seconds this takes 15."""
    if n < 3:
      return 1 if n else 0
    k = (n + 1) // 2 if n & 1 else n // 2
    k1, k2 = self.fibonacci_n_dp_cached_faster(k), self.fibonacci_n_dp_cached_faster(k - 1)
    return k1 * k1 + k2 * k2 if n & 1 else (2 * k2 + k1) * k1
