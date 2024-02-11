from math import isqrt


class MathAlgorithms:
    def __init__(self):
        self.primes_list = []
        self.primes_set = set()

    def sieve_of_eratosthenes(self, n_inclusive: int) -> None:
        """Generates list of primes up to n via eratosthenes method.

        Complexity: Time: O(n lnln(n)), Space: post call O(n/ln(n)), mid-call O(n)
        Variants: number and sum of prime factors, of diff prime factors, of divisors, and euler phi
        """
        limit, prime_sieve = isqrt(n_inclusive) + 1, [True] * (n_inclusive + 1)
        prime_sieve[0] = prime_sieve[1] = False
        for prime in range(2, limit):
            if prime_sieve[prime]:
                for composite in range(prime * prime, n_inclusive + 1, prime):
                    prime_sieve[composite] = False
        self.primes_list = [i for i, is_prime in enumerate(prime_sieve) if is_prime]

    def sieve_of_eratosthenes_optimized(self, n_inclusive: int) -> None:
        """Odds only optimized version of the previous method. Optimized to start at 3.

        Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space: post call O(n/ln(n)), mid-call O(n/2)
        """
        sqrt_n, limit = ((isqrt(n_inclusive) - 3) // 2) + 1, ((n_inclusive - 3) // 2) + 1
        primes_sieve = [True] * limit
        for i in range(sqrt_n):
            if primes_sieve[i]:
                prime = 2*i + 3
                start = (prime*prime - 3)//2
                for j in range(start, limit, prime):
                    primes_sieve[j] = False
        self.primes_list = [2] + [2*i + 3 for i, el in enumerate(primes_sieve) if el]

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
