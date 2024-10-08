from math import log, isqrt
from bisect import bisect_right
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
        limit += 1
        num_div = [1] * limit
        sum_div = [1] * limit
        for divisor in range(2, limit):
            for multiple in range(divisor, limit, divisor):
                num_div[multiple] += 1
                sum_div[multiple] += divisor
        self.num_divisors = num_div
        self.sum_divisors = sum_div

    def euler_phi_plus_sum_and_number_of_diff_prime_factors(self, limit: int) -> None:
        """This is basically same as sieve just using different ops. Complexity function 2."""
        limit += 1
        num_diff_pf = [0] * limit
        sum_diff_pf = [0] * limit
        phi = [i for i in range(limit)]
        for prime in range(2, limit):  # name is prime since we only iterate if its prime
            if num_diff_pf[prime] == 0:
                for multiple in range(prime, limit, prime):
                    num_diff_pf[multiple] += 1
                    sum_diff_pf[multiple] += prime
                    phi[multiple] = (phi[multiple] // prime) * (prime - 1)
        self.num_diff_prime_factors = num_diff_pf
        self.sum_diff_prime_factors = sum_diff_pf
        self.euler_phi = phi

    def num_and_sum_of_prime_factors(self, limit: int) -> None:
        """This uses similar idea to sieve but avoids divisions. Complexity function 3."""
        inclusive_limit = limit + 1
        num_pf = [0] * inclusive_limit
        sum_pf = [0] * inclusive_limit
        for prime in range(2, inclusive_limit):
            if num_pf[prime] == 0:  # or sum_pf if using that one
                exponent_limit = int(log(limit, prime)) + 2
                for exponent in range(1, exponent_limit):
                    prime_to_exponent = prime ** exponent
                    for multiple in range(prime_to_exponent, inclusive_limit, prime_to_exponent):
                        sum_pf[multiple] += prime
                        num_pf[multiple] += 1
        self.num_prime_factors = num_pf
        self.sum_prime_factors = sum_pf

    def num_and_sum_of_divisors_faster(self, limit: int) -> None:
        """runs in around x0.8-x0.5 the runtime of the slower one. Complexity function 4."""
        inclusive_limit = limit + 1
        num_divs = [1] * inclusive_limit
        sum_divs = [1] * inclusive_limit
        cur_pows = [1] * inclusive_limit
        for prime in range(2, inclusive_limit):
            if num_divs[prime] == 1:
                exponent_limit = int(log(limit, prime)) + 2
                for exponent in range(1, exponent_limit):
                    prime_to_exponent = prime ** exponent
                    for i in range(prime_to_exponent, inclusive_limit, prime_to_exponent):
                        cur_pows[i] += 1
                tmp = prime - 1  # this line and the line below used for sum_divs
                prime_powers = [prime ** exponent for exponent in range(0, exponent_limit + 1)]
                for multiple in range(prime, inclusive_limit, prime):
                    num_divs[multiple] *= cur_pows[multiple]
                    sum_divs[multiple] *= ((prime_powers[cur_pows[multiple]] - 1) // tmp)
                    cur_pows[multiple] = 1
        self.num_divisors = num_divs
        self.sum_divisors = sum_divs

    def prime_count_dp_slower(self, limit):
        root = isqrt(limit) + 1  # after this we mostly refer to root + 1 so just add one here
        end = limit // root
        phi_maybe = [i for i in range(1, end)] + [limit//i for i in range(root, 0, -1)]
        dp = [el for el in phi_maybe]
        dp_len, prime_index = len(dp), 0
        for prime in range(2, root):
            if dp[prime - 1] != dp[prime - 2]:
                prime_index += 1
                end = bisect_right(phi_maybe, prime * prime) - 2
                for i in range(dp_len - 1, end, -1):
                    k = phi_maybe[i] // prime
                    dp[i] -= dp[k - 1 if k < root else dp_len - (limit // k)] - prime_index
        return dp[dp_len - 1] - 1

    def prime_count_dp_faster(self, n: int):
        root, number_of_primes, step_multiplier = isqrt(n) + 1, n - 1, 1
        high, low, visited_sieve = [0] * root, [0] * root, [0] * root
        for divisor in range(2, root):
            low[divisor], high[divisor] = divisor - 1, n // divisor - 1
        for prime in range(2, root):
            if low[prime] != low[prime - 1]:  # happens only when n is prime
                t, end = low[prime - 1], min(root - 1, n // (prime * prime))
                number_of_primes -= high[prime] - t
                for i in range(prime + step_multiplier, end + 1, step_multiplier):
                    if not visited_sieve[i]:  # already visited this composite number
                        multiple = i * prime
                        is_above_root = multiple > root - 1
                        high[i] -= (low[n//multiple] if is_above_root else high[multiple]) - t
                for i in range(root - 1, prime * prime - 1, -1):
                    low[i] -= low[i // prime] - t
                for prime_multiple in range(prime * prime, end + 1, prime):
                    visited_sieve[prime_multiple] = 1
                if prime == 2:
                    step_multiplier += 1  # for, odds we can skip even numbers by stepping by 2
        return number_of_primes
