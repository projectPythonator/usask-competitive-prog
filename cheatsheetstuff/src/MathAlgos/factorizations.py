from math import gcd, isqrt
from collections import Counter
from itertools import takewhile
from typing import Dict


class MathAlgorithms:
    def __init__(self):
        self.min_primes_list = None
        self.primes_list = []
        self.primes_set = set()

    def sieve_of_eratosthenes_optimized(self, n_inclusive: int) -> None:
        """Odds only optimized version of the previous method. Optimized to start at 3.

        Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space: post call O(n/ln(n)), mid-call O(n/2)
        """
        sqrt_n, limit = ((isqrt(n_inclusive) - 3) // 2) + 1, ((n_inclusive - 3) // 2) + 1
        primes_sieve = [True] * limit
        for i in range(sqrt_n):
            if primes_sieve[i]:
                prime = 2 * i + 3
                start = (prime * prime - 3) // 2
                for j in range(start, limit, prime):
                    primes_sieve[j] = False
        self.primes_list = [2] + [2 * i + 3 for i, el in enumerate(primes_sieve) if el]

    def sieve_of_min_primes(self, n_inclusive: int) -> None:
        """Stores the min or max prime divisor for each number up to n.

        Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space: post call O(n)
        """
        min_primes = [0] * (n_inclusive + 1)
        min_primes[1] = 1
        for prime in reversed(self.primes_list):
            min_primes[prime] = prime
            start, end, step = prime * prime, n_inclusive + 1, prime if prime == 2 else 2 * prime
            for j in range(start, end, step):
                min_primes[j] = prime
        self.min_primes_list = min_primes

    def prime_factorize_n_trivial(self, n: int) -> Dict:
        limit, prime_factors = isqrt(n) + 1, []
        for prime in range(2, limit):
            while n % prime == 0:
                n //= prime
                prime_factors.append(prime)
        if n > 1:  # n is prime or last factor of n is prime
            prime_factors.append(n)
        return Counter(prime_factors)

    def prime_factorize_n(self, n: int) -> Dict:  # using this for testing
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
        return Counter(prime_factors)

    def prime_factorize_n_log_n(self, n: int) -> Dict:
        """An optimized prime factorization of n function based on min primes already sieved.

        Complexity: Time: O(log n), Space: O(log n)
        Optimization: assign append to function and assign min_prime[n] to a value in the loop
        """
        prime_factors = []
        while n > 1:
            prime_factors.append(self.min_primes_list[n])
            n = n // self.min_primes_list[n]
        return Counter(prime_factors)
    def polynomial_function_f(self, x: int, c: int, m: int) -> int:
        """Represents the function f(x) = (x^2 + c) in pollard rho and brent, cycle finding."""
        return (x * x + c) % m  # paste this in code for speed up. is here for clarity only

    def pollard_rho(self, n: int, x0=2, c=1) -> int:
        """Semi fast integer factorization. Based on the birthday paradox and floyd cycle finding.

        Complexity per call: Time: O(min(max(p), n^0.25) * ln n), Space: O(log2(n) bits)
        """
        x, y, g = x0, x0, 1
        while g == 1:  # when g != 1 then we found a divisor of n shared with x - y
            x = self.polynomial_function_f(x, c, n)
            y = self.polynomial_function_f(self.polynomial_function_f(y, c, n), c, n)
            g = gcd(abs(x - y), n)
        return g

    def brent_pollard_rho(self, n: int, x0=2, c=1) -> int:
        """Faster version of above. Similar time complexity. uses faster cycle finder."""
        x, m = x0, 128  # 128 here is used as a small power of 2 vs using 100 more below
        g = q = left = 1
        xs = y = 0
        while g == 1:
            y, k = x, 0
            for _ in range(1, left):
                x = (x * x + c) % n
            while k < left and g == 1:
                xs, end = x, min(m, left - k)   # here we are using a technique similar to cache
                for _ in range(end):            # and loop unrolling were we try for sets of cycles
                    x = (x * x + c) % n         # if we over shoot we can just go back which is
                    q = (q * abs(y - x)) % n    # technically what end computes
                k, g = k + m, gcd(q, n)
            left = left * 2
        if g == n:
            while True:
                xs = (xs * xs + c) % n
                g = gcd(abs(xs - y), n)
                if g != 1:
                    break
        return g