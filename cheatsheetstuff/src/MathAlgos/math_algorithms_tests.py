import unittest
from random import randint
import primality_testing as prime_tests
import prime_sieves
import prime_sieve_variants
import factorizations
import fibonacci as fib
import catalan_functions
import binomial_coefficient


class TestMathMethods(unittest.TestCase):
    def test_is_prime_optimized_up_to_100(self):
        primes_to_100 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                         43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        obj = prime_tests.MathAlgorithms()
        obj.primes_list = [2, 3, 5, 7, 11]
        obj.primes_set = set(obj.primes_list)
        for number in range(100):
            if obj.is_prime_optimized(number):
                self.assertTrue(number in primes_to_100)
            else:
                self.assertTrue(number not in primes_to_100)

    def test_is_prime_optimized_vs_sieve_up_to_n(self):
        limit = 1000000
        test_limit = 2000
        obj = prime_tests.MathAlgorithms()
        test_obj = prime_tests.MathAlgorithms()
        obj.sieve_of_eratosthenes(limit)
        test_obj.sieve_of_eratosthenes(test_limit)
        test_obj.primes_set = set(test_obj.primes_list)
        primes_generated_by_sieve = set(obj.primes_list)
        for number in range(limit):
            if test_obj.is_prime_optimized(number):
                self.assertTrue(number in primes_generated_by_sieve)
            else:
                self.assertTrue(number not in primes_generated_by_sieve)

    def test_is_prime_trivial_up_to_1m(self):
        limit = 1000000
        obj = prime_tests.MathAlgorithms()
        obj.sieve_of_eratosthenes(limit)
        obj.primes_set = set(obj.primes_list)
        for number in range(limit):
            self.assertEqual(obj.is_prime_optimized(number), obj.is_prime_trivial(number))

    def test_miller_rabin_primality_test_1m(self):
        test_limit = 1000000
        limit = 1000000
        random_test_limit = 10
        test_obj = prime_tests.MathAlgorithms()
        test_obj.sieve_of_eratosthenes(test_limit)
        test_obj.primes_set = set(test_obj.primes_list)
        obj = prime_tests.MathAlgorithms()
        obj.miller_rabin_primality_test_prep()
        for number in range(limit):
            self.assertEqual(test_obj.is_prime_optimized(number),
                             obj.miller_rabin_primality_test(number))
        for _ in range(random_test_limit):
            number = randint(2 ** 50, 2 ** 55)
            self.assertEqual(test_obj.is_prime_trivial(number),
                             obj.miller_rabin_primality_test(number), number)

    def test_sieve_of_eratosthenes_100(self):
        primes_to_100 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                         43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        obj = prime_sieves.MathAlgorithms()
        obj.sieve_of_eratosthenes(100)
        self.assertEqual(obj.primes_list, primes_to_100)

    def test_sieve_of_eratosthenes_1m(self):
        limit = 1000000
        test_obj = prime_tests.MathAlgorithms()
        obj = prime_sieves.MathAlgorithms()
        obj.sieve_of_eratosthenes(limit)
        obj.primes_set = set(obj.primes_list)
        for number in range(limit):
            if test_obj.is_prime_trivial(number):
                self.assertTrue(number in obj.primes_set)
            else:
                self.assertTrue(number not in obj.primes_set)

    def test_sieve_of_eratosthenes_optimized_1m(self):
        limit = 1000000
        test_obj = prime_sieves.MathAlgorithms()
        obj = prime_sieves.MathAlgorithms()
        test_obj.sieve_of_eratosthenes(limit)
        obj.sieve_of_eratosthenes_optimized(limit)
        self.assertEqual(test_obj.primes_list, obj.primes_list)

    def testing_sieve_of_min_primes_1m(self):
        """Depends on sieve_of_eratosthenes_optimized and prime_factorize_n working."""
        limit = 1000000
        obj = prime_sieve_variants.MathAlgorithms()
        obj.sieve_of_eratosthenes_optimized(limit)
        obj.sieve_of_min_primes(limit)
        for i in range(2, limit):
            self.assertEqual(obj.min_primes_list[i], min(obj.prime_factorize_n(i)))

    def testing_num_and_sum_of_divisors_faster_1m(self):
        """Depends on sieve_of_eratosthenes_optimized and prime_factorize_n working."""
        limit = 1000000
        obj = prime_sieve_variants.MathAlgorithms()
        obj.sieve_of_eratosthenes_optimized(limit)
        obj.num_and_sum_of_divisors_faster(limit)
        for i, (num_div, sum_div) in enumerate(zip(obj.num_divisors, obj.sum_divisors)):
            if i < 2:
                continue
            factors = obj.prime_factorize_n(i)
            expected_num = expected_sum = 1
            for prime, power in factors.items():
                expected_num *= (power + 1)
                expected_sum *= ((prime ** (power + 1) - 1) // (prime - 1))
            self.assertEqual(expected_sum, sum_div)
            self.assertEqual(expected_num, num_div)

    def testing_num_and_sum_of_divisors_same_as_faster_version_1m(self):
        """Depends on sieve_of_eratosthenes_optimized and prime_factorize_n working."""
        limit = 1000000
        obj1 = prime_sieve_variants.MathAlgorithms()
        obj2 = prime_sieve_variants.MathAlgorithms()
        obj1.num_and_sum_of_divisors(limit)
        obj2.num_and_sum_of_divisors_faster(limit)
        self.assertEqual(obj1.num_divisors, obj2.num_divisors)
        self.assertEqual(obj1.sum_divisors, obj2.sum_divisors)

    def testing_euler_phi_plus_sum_and_number_of_diff_prime_factors_1m(self):
        """Depends on sieve_of_eratosthenes_optimized and prime_factorize_n working."""
        limit = 1000000
        obj = prime_sieve_variants.MathAlgorithms()
        obj.sieve_of_eratosthenes_optimized(limit)
        obj.euler_phi_plus_sum_and_number_of_diff_prime_factors(limit)
        for i, (num_diff_pf, sum_diff_pf, phi) in enumerate(zip(obj.num_diff_prime_factors,
                                                                obj.sum_diff_prime_factors,
                                                                obj.euler_phi)):
            if i < 2:
                continue
            factors = obj.prime_factorize_n(i)
            expected_phi = i
            for prime in factors:
                expected_phi -= (expected_phi // prime)
            self.assertEqual(len(factors), num_diff_pf, i)
            self.assertEqual(sum(factors), sum_diff_pf, i)
            self.assertEqual(expected_phi, phi)

    def testing_num_and_sum_of_prime_factors_1m(self):
        """Depends on sieve_of_eratosthenes_optimized and prime_factorize_n working."""
        limit = 1000000
        obj = prime_sieve_variants.MathAlgorithms()
        obj.sieve_of_eratosthenes_optimized(limit)
        obj.num_and_sum_of_prime_factors(limit)
        for i, (num_pf, sum_pf) in enumerate(zip(obj.num_prime_factors, obj.sum_prime_factors)):
            if i < 2:
                continue
            factors = obj.prime_factorize_n(i)
            self.assertEqual(sum(factors.values()), num_pf, i)
            self.assertEqual(sum(prime * power for prime, power in factors.items()), sum_pf, i)

    def test_prime_factorize_n_against_prime_factorize_n_trivial_1m(self):
        """Assumes prime_factorize_n_trivial works."""
        limit = 1000000
        obj = factorizations.MathAlgorithms()
        obj.sieve_of_eratosthenes_optimized(limit)
        for number in range(2, limit):
            expected_factors = obj.prime_factorize_n_trivial(number)
            result_factors = obj.prime_factorize_n(number)
            self.assertEqual(len(expected_factors), len(result_factors))
            for prime, power in result_factors.items():
                self.assertTrue(prime in expected_factors)
                self.assertEqual(power, expected_factors[prime])

    def test_prime_factorize_n_log_n_1m(self):
        """Assumes prime_factorize_n_trivial works."""
        limit = 1000000
        obj = factorizations.MathAlgorithms()
        obj.sieve_of_eratosthenes_optimized(limit)
        obj.sieve_of_min_primes(limit)
        for number in range(2, limit):
            expected_factors = obj.prime_factorize_n(number)
            result_factors = obj.prime_factorize_n_log_n(number)
            self.assertEqual(len(expected_factors), len(result_factors))
            for prime, power in result_factors.items():
                self.assertTrue(prime in expected_factors)
                self.assertEqual(power, expected_factors[prime])

    def test_prime_factorize_n_variants_1m(self):
        limit = 1000000
        obj_base = prime_sieve_variants.MathAlgorithms()
        obj = factorizations.MathAlgorithms()
        obj_base.num_and_sum_of_divisors_faster(limit)
        obj_base.num_and_sum_of_prime_factors(limit)
        obj_base.euler_phi_plus_sum_and_number_of_diff_prime_factors(limit)
        obj.sieve_of_eratosthenes_optimized(limit)
        for i, expected_tuple in enumerate(zip(obj_base.num_diff_prime_factors,
                                               obj_base.sum_diff_prime_factors,
                                               obj_base.num_prime_factors,
                                               obj_base.sum_prime_factors,
                                               obj_base.num_divisors,
                                               obj_base.sum_divisors,
                                               obj_base.euler_phi)):
            if i < 2:
                continue
            result_tuple = obj.prime_factorize_n_variants(i)
            self.assertEqual(result_tuple, expected_tuple)

    def test_block_sieve_odd_10k(self):
        limit = 10000
        start = 1000
        obj_base = prime_sieves.MathAlgorithms()
        obj = prime_sieves.MathAlgorithms()
        for i in range(start, start+limit):
            obj_base.sieve_of_eratosthenes_optimized(i)
            obj.block_sieve_odd(i)
            self.assertEqual(len(obj_base.primes_list), len(obj.primes_list), i)
            self.assertEqual(obj_base.primes_list[-1], obj.primes_list[-1], i)

    def test_fibonacci_n_iterative_10(self):
        limit = 10
        obj = fib.MathAlgorithms()
        obj.fibonacci_n_iterative(limit)
        self.assertEqual([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55], obj.fibonacci_list)

    def test_fibonacci_n_dp_cached_faster_10k(self):
        limit = 10000
        obj = fib.MathAlgorithms()
        obj.fibonacci_n_iterative(limit)
        for i in range(limit):
            self.assertEqual(obj.fibonacci_n_dp_cached_faster(i), obj.fibonacci_list[i], i)

    def test_fibonacci_n_dp_cached_10k_or_1m(self):
        limit = 10000
        limit_higher = 10000000
        obj = fib.MathAlgorithms()
        fib_set_1 = [obj.fibonacci_n_dp_cached(i) for i in range(limit)]
        fib_set_2 = [obj.fibonacci_n_dp_cached_faster(i) for i in range(limit)]
        self.assertEqual(fib_set_1, fib_set_2)

    def test_generate_catalan_n_31(self):
        expected = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900,
                    2674440, 9694845, 35357670, 129644790, 477638700, 1767263190, 6564120420,
                    24466267020, 91482563640, 343059613650, 1289904147324, 4861946401452,
                    18367353072152, 69533550916004, 263747951750360, 1002242216651368,
                    3814986502092304]
        limit = len(expected)
        obj = catalan_functions.MathAlgorithms()
        obj.generate_catalan_n(limit)
        self.assertEqual(obj.catalan_numbers, expected)

    def test_generate_catalan_n_mod_inverse_31(self):
        expected = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900,
                    2674440, 9694845, 35357670, 129644790, 477638700, 1767263190, 6564120420,
                    24466267020, 91482563640, 343059613650, 1289904147324, 4861946401452,
                    18367353072152, 69533550916004, 263747951750360, 1002242216651368,
                    3814986502092304]
        mod_m = 10**9+7
        expected = [num % mod_m for num in expected]
        limit = len(expected)
        obj = catalan_functions.MathAlgorithms()
        obj.generate_catalan_n_mod_inverse(limit, mod_m)
        self.assertEqual(obj.catalan_numbers, expected)

    def test_catalan_via_prime_factors_faster_10k(self):
        """was tested on 100k but keeping it on 10k for default"""
        mod_m = 10**9+7
        limit = 10000
        test_obj = catalan_functions.MathAlgorithms()
        obj = catalan_functions.MathAlgorithms()
        test_obj.generate_catalan_n_mod_inverse(limit, mod_m)
        for n in range(1, limit):
            self.assertEqual(obj.catalan_via_prime_factors_faster(2*n, n, mod_m),
                             test_obj.catalan_numbers[n])

    def test_catalan_via_prime_factors_slower_10k(self):
        mod_m = 10**9+7
        limit = 10000
        test_obj = catalan_functions.MathAlgorithms()
        obj = catalan_functions.MathAlgorithms()
        test_obj.generate_catalan_n_mod_inverse(limit, mod_m)
        for n in range(1, limit):
            self.assertEqual(obj.catalan_via_prime_factors_slower(2*n, n, mod_m),
                             test_obj.catalan_numbers[n])

    def test_catalan_via_prime_factors_fast_and_slow_random_n_1m_100m(self):
        mod_m = 10**9+7
        limit = randint(1000000, 100000000)
        test_obj = catalan_functions.MathAlgorithms()
        obj = catalan_functions.MathAlgorithms()
        self.assertEqual(test_obj.catalan_via_prime_factors_slower(2 * limit, limit, mod_m),
                         obj.catalan_via_prime_factors_faster(2 * limit, limit, mod_m))

    def test_binomial_coefficient_up_to_n_8(self):
        pascal = [[1],
                  [1, 1],
                  [1, 2, 1],
                  [1, 3, 3, 1],
                  [1, 4, 6, 4, 1],
                  [1, 5, 10, 10, 5, 1],
                  [1, 6, 15, 20, 15, 6, 1],
                  [1, 7, 21, 35, 35, 21, 7, 1],
                  [1, 8, 28, 56, 70, 56, 28, 8, 1]]
        obj = binomial_coefficient.MathAlgorithms()
        for n, row in enumerate(pascal):
            for k, expected in enumerate(row):
                result = obj.binomial_coefficient_dp_with_cache(n, k)
                self.assertEqual(result, expected)

    def test_binomial_coefficient_n_mod_p_prep_and_c_n_k_up_to_100(self):
        limit = 1000
        mod_m = 10**9+7
        inverse = False
        test_obj = binomial_coefficient.MathAlgorithms()
        obj = binomial_coefficient.MathAlgorithms()
        obj.binomial_coefficient_n_mod_p_prep(limit+1, mod_m, inverse)
        for n in range(limit):
            for k in range(n+1):
                expected = test_obj.binomial_coefficient_dp_with_cache(n, k) % mod_m
                result = obj.c_n_k(n, k, inverse)
                self.assertEqual(result, expected, "{} {}".format(n, k))

