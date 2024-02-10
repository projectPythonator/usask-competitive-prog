from math import isqrt, log


class MathAlgorithms:
    def __init__(self):
        self.primes_list = []
        self.primes_set = set()

        self.min_primes_list = []
        self.sum_prime_factors = []
        self.num_divisors = []
        self.sum_divisors = []
        self.euler_phi = []
        self.sum_diff_prime_factors = []
        self.num_diff_prime_factors = []
        self.num_prime_factors = []

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

    def sieve_of_eratosthenes_variants(self, n_inclusive: int) -> None:
        """Seven variants of prime sieve listed above.

        Complexity:
            function 1: Time: O(n lnln(n)), Space: O(n)
            function 3: Time: O(n log(n)), Space: O(n)
            function 2: Time: O(n lnln(n) log(n)), Space: O(n)
        """
        self.euler_phi_plus_sum_and_number_of_diff_prime_factors(n_inclusive)
        self.num_and_sum_of_divisors(n_inclusive)
        self.num_and_sum_of_prime_factors(n_inclusive)

    def euler_phi_plus_sum_and_number_of_diff_prime_factors(self, limit: int) -> None:
        """This is basically same as sieve just using different ops. Complexity function 1."""
        num_diff_pf = [0] * (limit + 1)
        sum_diff_pf = [0] * (limit + 1)
        phi = [i for i in range(limit + 1)]
        for i in range(2, limit):
            if num_diff_pf[i] == 0:
                for j in range(i, limit, i):
                    num_diff_pf[j] += 1
                    sum_diff_pf[j] += i
                    phi[j] = (phi[j]//i) * (i-1)
        self.num_diff_prime_factors = num_diff_pf
        self.sum_diff_prime_factors = sum_diff_pf
        self.euler_phi = phi

    def num_and_sum_of_divisors(self, limit: int) -> None:
        """Does a basic sieve. Complexity function 2."""
        num_div = [1] * (limit + 1)
        sum_div = [1] * (limit + 1)
        for i in range(2, limit):
            for j in range(i, limit, i):
                num_div[j] += 1
                sum_div[j] += i
        self.num_divisors = num_div
        self.sum_divisors = sum_div

    def num_and_sum_of_prime_factors(self, limit: int) -> None:
        """This uses similar idea to sieve but avoids divisions. Complexity function 3."""
        num_pf = [0] * (limit + 1)
        sum_pf = [0] * (limit + 1)
        for prime in self.primes_list:
            exponent_limit = int(log(limit, prime)) + 1
            for exponent in range(1, exponent_limit):
                prime_to_exponent = prime**exponent
                for i in range(prime_to_exponent, limit + 1, prime_to_exponent):
                    sum_pf[i] += prime
                    num_pf[i] += 1
        self.num_prime_factors = num_pf
        self.sum_prime_factors = sum_pf

