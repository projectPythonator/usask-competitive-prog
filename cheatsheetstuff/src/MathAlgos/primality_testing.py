


class MathAlgorithms:
    def is_prime_trivial(self, n: int) -> bool:
        """Tests if n is prime via divisors up to sqrt(n).

        Complexity per call: Time: O(sqrt(n)), T(sqrt(n)/3), Space: O(1)
        Optimizations: 6k + i method, since we checked 2 and 3 only need test form 6k + 1 and 6k + 5
        """
        if n < 4:  # base case of n in 0, 1, 2, 3
            return n > 1
        if n % 2 == 0 or n % 3 == 0:  # this check is what allows us to use 6k + i
            return False
        limit = isqrt(n) + 1
        for p in range(5, limit, 6):
            if n % p == 0 or n % (p+2) == 0:
                return False
        return True