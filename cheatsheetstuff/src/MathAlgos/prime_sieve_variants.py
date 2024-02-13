from math import log
import prime_sieves


class MathAlgorithms:
    def __init__(self):
        self.min_primes_list = []
        self.sum_prime_factors = []
        self.num_divisors = []
        self.sum_divisors = []
        self.euler_phi = []
        self.sum_diff_prime_factors = []
        self.num_diff_prime_factors = []
        self.num_prime_factors = []

        self.sieve_obj = prime_sieves.MathAlgorithms()
        self.sieve_function = self.sieve_obj.prime_sieve_super_fast
        self.primes_list = []

    def fill_primes_list_and_set(self, n):
        """Fills primes list using sieve function"""
        self.sieve_function(n)
        self.primes_list = self.sieve_obj.primes_list

    def sieve_of_min_primes(self, n_inclusive: int) -> None:
        """Stores the min or max prime divisor for each number up to n.

        Complexity: Time: O(max(n lnln(sqrt(n)), n)), Space: post call O(n)
        """
        self.fill_primes_list_and_set(n_inclusive)
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
            function 1: Time: O(n log(n)), Space: O(n)
            function 2: Time: O(n lnln(n)), Space: O(n)
            function 3: Time: O(n lnln(n) log(n)), Space: O(n)
            function 4: Time: O(n lnln(n) log(n)), Space: O(n)
        """
        self.euler_phi_plus_sum_and_number_of_diff_prime_factors(n_inclusive)
        self.num_and_sum_of_divisors(n_inclusive)
        self.num_and_sum_of_prime_factors(n_inclusive)

    def num_and_sum_of_divisors(self, limit: int) -> None:
        """Does a basic sieve. Complexity function 1."""
        num_div = [1] * (limit + 1)
        sum_div = [1] * (limit + 1)
        for i in range(2, limit + 1):
            for j in range(i, limit + 1, i):
                num_div[j] += 1
                sum_div[j] += i
        self.num_divisors = num_div
        self.sum_divisors = sum_div

    def euler_phi_plus_sum_and_number_of_diff_prime_factors(self, limit: int) -> None:
        """This is basically same as sieve just using different ops. Complexity function 2."""
        num_diff_pf = [0] * (limit + 1)
        sum_diff_pf = [0] * (limit + 1)
        phi = [i for i in range(limit + 1)]
        for i in range(2, limit + 1):
            if num_diff_pf[i] == 0:
                for j in range(i, limit + 1, i):
                    num_diff_pf[j] += 1
                    sum_diff_pf[j] += i
                    phi[j] = (phi[j]//i) * (i-1)
        self.num_diff_prime_factors = num_diff_pf
        self.sum_diff_prime_factors = sum_diff_pf
        self.euler_phi = phi

    def num_and_sum_of_prime_factors(self, limit: int) -> None:
        """This uses similar idea to sieve but avoids divisions. Complexity function 3."""
        num_pf = [0] * (limit + 1)
        sum_pf = [0] * (limit + 1)
        for prime in range(2, limit + 1):
            if num_pf[prime] == 0:  # or sum_pf if using that one
                exponent_limit = int(log(limit, prime)) + 2
                for exponent in range(1, exponent_limit):
                    prime_to_exponent = prime**exponent
                    for i in range(prime_to_exponent, limit + 1, prime_to_exponent):
                        sum_pf[i] += prime
                        num_pf[i] += 1
        self.num_prime_factors = num_pf
        self.sum_prime_factors = sum_pf

    def num_and_sum_of_divisors_faster(self, limit: int) -> None:
        """runs in around x0.8-x0.5 the runtime of the slower one. Complexity function 4."""
        num_divs = [1] * (limit + 1)
        sum_divs = [1] * (limit + 1)
        cur_pows = [1] * (limit + 1)
        for prime in range(2, limit + 1):
            if num_divs[prime] == 1:
                exponent_limit = int(log(limit, prime)) + 2
                for exponent in range(1, exponent_limit):
                    prime_to_exponent = prime ** exponent
                    for i in range(prime_to_exponent, limit + 1, prime_to_exponent):
                        cur_pows[i] += 1
                tmp = prime - 1  # this line and the line below used for sum_divs
                prime_powers = [prime ** exponent for exponent in range(0, exponent_limit+1)]
                for i in range(prime, limit + 1, prime):
                    num_divs[i] *= cur_pows[i]
                    sum_divs[i] *= ((prime_powers[cur_pows[i]] - 1) // tmp)
                    cur_pows[i] = 1
        self.num_divisors = num_divs
        self.sum_divisors = sum_divs
