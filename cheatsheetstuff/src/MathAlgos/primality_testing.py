from math import isqrt
from itertools import takewhile
from random import choices


class MathAlgorithms:
    def __init__(self):
        self.primes_list = []
        self.mrpt_known_bounds = []
        self.mrpt_known_tests = []
        self.primes_set = set()

    def sieve_of_eratosthenes(self, limit: int) -> None:
        """Odds only optimized version of the previous method. Optimized to start at 3.

        Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space: post call O(n/ln(n)), mid-call O(n/2)
        """
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

    def is_prime_optimized(self, n):
        if n < self.primes_list[-1]:
            return n in self.primes_set
        limit = isqrt(n) + 1
        for prime in takewhile(lambda x: x < limit, self.primes_list):
            if n % prime == 0:
                return False
        return True

    def is_composite(self, a: int, d: int, n: int, s: int) -> bool:
        """The witness test of miller rabin.

        Complexity per call: Time O(log^3(n)), Space: O(2**s, bits)
        """
        if 1 == pow(a, d, n):
            return False
        for i in range(s):
            if n-1 == pow(a, d * 2**i, n):
                return False
        return True

    def miller_rabin_primality_test(self, n: int, precision_for_huge_n=16) -> bool:
        """Probabilistic primality test with error rate of 4^(-k) past 341550071728321.

        Complexity per call: Time O(k log^3(n)), Space: O(2**s) bits
        Note: range(16) used to just do a small test to weed out lots of numbers.
        """
        if n < self.primes_list[-1]:
            return n in self.primes_set
        if any((n % self.primes_list[p] == 0) for p in range(32)) or n == 3215031751:
            return False  # 3215031751 is an edge case for this data set
        d, s = n-1, 0
        while d % 2 == 0:
            d, s = d//2, s+1
        if n < self.mrpt_known_bounds[-1]:
            for i, bound in enumerate(self.mrpt_known_bounds, 2):
                if n < bound:
                    return not any(self.is_composite(self.mrpt_known_tests[j], d, n, s)
                                   for j in range(i))
        return not any(self.is_composite(prime, d, n, s)
                       for prime in choices(self.primes_list, k=precision_for_huge_n))

    def miller_rabin_primality_test_prep(self):
        """This function needs to be called before miller rabin"""
        self.mrpt_known_bounds = [1373653, 25326001, 118670087467,
                                  2152302898747, 3474749660383, 341550071728321]
        self.mrpt_known_tests = [2, 3, 5, 7, 11, 13, 17]
        self.sieve_of_eratosthenes(1000)         # comment out if different size needed
        self.primes_set = set(self.primes_list)  # comment out if already have bigger size