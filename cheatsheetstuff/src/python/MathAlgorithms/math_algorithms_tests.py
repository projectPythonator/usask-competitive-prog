import unittest
from random import randint
import primality_testing as prime_tests
import prime_sieves
import prime_sieve_variants
import factorizations
import fibonacci as fib
import catalan_functions
import binomial_coefficient
import fast_fourier_transform
import chinese_remainder_theorem
from bisect import bisect_right
import gc


class TestMathMethods(unittest.TestCase):
    """Pypy runs the tests pretty fast."""
    #
    # @classmethod
    # def setUpClass(cls):
    #     gc.collect()

    @classmethod
    def tearDownClass(cls):
        gc.collect()
    
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
        obj.fill_primes_list_and_set(limit)
        test_obj.fill_primes_list_and_set(test_limit)
        for number in range(limit):
            if test_obj.is_prime_optimized(number):
                self.assertTrue(number in obj.primes_set)
            else:
                self.assertTrue(number not in obj.primes_set)

    def test_is_prime_trivial_up_to_1m(self):
        limit = 1000000
        obj = prime_tests.MathAlgorithms()
        obj.fill_primes_list_and_set(limit)
        for number in range(limit):
            self.assertEqual(obj.is_prime_optimized(number), obj.is_prime_trivial(number))

    def test_miller_rabin_primality_test_1m(self):
        test_limit = 1000000
        limit = 1000000
        random_test_limit = 10
        base_obj = prime_tests.MathAlgorithms()
        base_obj.fill_primes_list_and_set(test_limit)
        obj = prime_tests.MathAlgorithms()
        obj.miller_rabin_primality_test_prep()
        for number in range(limit):
            self.assertEqual(base_obj.is_prime_optimized(number),
                             obj.miller_rabin_primality_test(number))
        for _ in range(random_test_limit):
            number = randint(2 ** 50, 2 ** 55)
            self.assertEqual(base_obj.is_prime_trivial(number),
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

    def test_prime_sieve_super_fast_10k_runs(self):
        """tested up to 100k to match sieves."""
        limit = 10000
        obj1 = prime_sieves.MathAlgorithms()
        obj2 = prime_sieves.MathAlgorithms()
        obj2.sieve_of_eratosthenes(limit+1)
        for i in range(10, limit):
            obj1.prime_sieve_super_fast(i)
            self.assertEqual(obj1.primes_list,
                             obj2.primes_list[:bisect_right(obj2.primes_list,
                                                            obj1.primes_list[-1])])

    def test_sieve_of_eratosthenes_optimized_10k_runs(self):
        """tested up to 100k to match sieves."""
        limit = 10000
        obj1 = prime_sieves.MathAlgorithms()
        obj2 = prime_sieves.MathAlgorithms()
        obj2.prime_sieve_super_fast(limit+1)
        for i in range(10, limit):
            obj1.sieve_of_eratosthenes_optimized(i)
            self.assertEqual(obj1.primes_list,
                             obj2.primes_list[:bisect_right(obj2.primes_list,
                                                            obj1.primes_list[-1])])

    def test_block_sieve_odd_10k_runs(self):
        """tested up to 100k to match sieves."""
        limit = 10000
        obj1 = prime_sieves.MathAlgorithms()
        obj2 = prime_sieves.MathAlgorithms()
        obj2.prime_sieve_super_fast(limit+1)
        for i in range(10, limit):
            obj1.block_sieve_odd(i)
            self.assertEqual(obj1.primes_list,
                             obj2.primes_list[:bisect_right(obj2.primes_list,
                                                            obj1.primes_list[-1])])

    def test_all_4_sieves_on_powers_of_10_up_to_10m(self):
        """tested up to 100m before."""
        limit = 8
        obj1 = prime_sieves.MathAlgorithms()
        obj2 = prime_sieves.MathAlgorithms()
        obj3 = prime_sieves.MathAlgorithms()
        obj4 = prime_sieves.MathAlgorithms()
        obj5 = prime_sieves.MathAlgorithms()
        for power in range(1, limit):
            n_limit = 10 ** power
            obj1.sieve_of_eratosthenes(n_limit)
            obj2.sieve_of_eratosthenes_optimized(n_limit)
            obj3.block_sieve_odd(n_limit)
            obj4.prime_sieve_super_fast(n_limit)
            obj5.prime_sieve_super_fast_faster_maybe(n_limit)
            self.assertEqual(obj1.primes_list, obj2.primes_list)
            self.assertEqual(obj1.primes_list, obj3.primes_list)
            self.assertEqual(obj1.primes_list, obj4.primes_list)
            self.assertEqual(obj1.primes_list, obj5.primes_list)

    def test_all_4_sieves_on_powers_of_2_up_to_100m(self):
        """tested up to 100m before."""
        limit = 24
        obj1 = prime_sieves.MathAlgorithms()
        obj2 = prime_sieves.MathAlgorithms()
        obj3 = prime_sieves.MathAlgorithms()
        obj4 = prime_sieves.MathAlgorithms()
        obj5 = prime_sieves.MathAlgorithms()
        for power in range(1, limit):
            n_limit = 2 ** power
            obj1.sieve_of_eratosthenes(n_limit)
            obj2.sieve_of_eratosthenes_optimized(n_limit)
            obj3.block_sieve_odd(n_limit)
            obj4.prime_sieve_super_fast(n_limit)
            obj5.prime_sieve_super_fast_faster_maybe(n_limit)
            self.assertEqual(obj1.primes_list, obj2.primes_list)
            self.assertEqual(obj1.primes_list, obj3.primes_list)
            self.assertEqual(obj1.primes_list, obj4.primes_list)
            self.assertEqual(obj1.primes_list, obj5.primes_list)

    def testing_prime_count_dp_faster_100k(self):
        """tested up to 1m"""
        limit = 10**5
        obj2 = prime_sieve_variants.MathAlgorithms()
        obj2.fill_primes_list_and_set(limit)
        for number in range(10, limit):
            ind = bisect_right(obj2.primes_list, number)
            result = obj2.prime_count_dp_faster(number)
            self.assertEqual(result, ind)

    def testing_prime_count_dp_slower_100k(self):
        """tested up to 1m"""
        limit = 10**5
        obj2 = prime_sieve_variants.MathAlgorithms()
        obj2.fill_primes_list_and_set(limit)
        for number in range(10, limit):
            ind = bisect_right(obj2.primes_list, number)
            result = obj2.prime_count_dp_slower(number)
            self.assertEqual(result, ind)

    def testing_prime_count_dp_faster_and_slower_powers_10_up_to_12(self):
        limit_power = 12
        limit_base = 10
        expected = [0, 4, 25, 168, 1229, 9592, 78498, 664579, 5761455, 50847534, 455052511,
                    4118054813, 37607912018]
        obj = prime_sieve_variants.MathAlgorithms()
        for power in range(1, limit_power):
            limit = limit_base ** power
            result_1 = obj.prime_count_dp_slower(limit)
            result_2 = obj.prime_count_dp_slower(limit)
            self.assertEqual(result_1, expected[power])
            self.assertEqual(result_2, expected[power])

    def testing_prime_count_dp_faster_and_slower_powers_2_up_to_37(self):
        limit_power = 37
        limit_base = 2
        expected = [0, 1, 2, 4, 6, 11, 18, 31, 54, 97, 172, 309, 564, 1028, 1900, 3512, 6542,
                    12251, 23000, 43390, 82025, 155611, 295947, 564163, 1077871, 2063689,
                    3957809, 7603553, 14630843, 28192750, 54400028, 105097565, 203280221,
                    393615806, 762939111, 1480206279, 2874398515, 5586502348, 10866266172,
                    21151907950, 41203088796]
        obj = prime_sieve_variants.MathAlgorithms()
        for power in range(1, limit_power):
            limit = limit_base ** power
            result_1 = obj.prime_count_dp_slower(limit)
            result_2 = obj.prime_count_dp_slower(limit)
            self.assertEqual(result_1, expected[power])
            self.assertEqual(result_2, expected[power])

    def testing_sieve_of_min_primes_10k_runs(self):
        """tested on 100k runs"""
        limit = 10000
        factor_obj = factorizations.MathAlgorithms()
        obj = prime_sieve_variants.MathAlgorithms()
        expected = [0, 1] + [min(factor_obj.prime_factorize_n_trivial(i))
                             for i in range(2, limit + 10)]
        for i in range(2, limit):
            obj.sieve_of_min_primes(i)
            self.assertEqual(obj.min_primes_list,
                             expected[:len(obj.min_primes_list)], i)

    def testing_sieve_of_min_primes_on_powers_2_and_10_up_to_100k(self):
        """tested up to 1m"""
        limit_2 = 2**20
        power_10 = 6
        power_2 = 20
        factor_obj = factorizations.MathAlgorithms()
        factor_obj.fill_primes_list_and_set(limit_2)
        expected = [0, 1] + [min(factor_obj.prime_factorize_n(i))
                             for i in range(2, limit_2)]
        obj = prime_sieve_variants.MathAlgorithms()
        for i in range(1, power_10):
            limit = 10 ** i
            obj.sieve_of_min_primes(limit)
            self.assertEqual(obj.min_primes_list,
                             expected[:limit+1])
        for i in range(1, power_2):
            limit = 2 ** i
            obj.sieve_of_min_primes(limit)
            self.assertEqual(obj.min_primes_list,
                             expected[:limit+1])

    def testing_num_and_sum_of_divisors_faster_5k_runs(self):
        """tested up to 30k"""
        limit = 5000
        test_obj = factorizations.MathAlgorithms()
        obj = prime_sieve_variants.MathAlgorithms()
        test_obj.fill_min_primes_list(limit)
        expected = [(1, 1), (1, 1)]
        for i in range(2, limit):
            factors = test_obj.prime_factorize_n_log_n(i)
            expected_num = expected_sum = 1
            for prime, power in factors.items():
                expected_num *= (power + 1)
                expected_sum *= ((prime ** (power + 1) - 1) // (prime - 1))
            expected.append((expected_num, expected_sum))
        for i in range(2, limit):
            obj.num_and_sum_of_divisors_faster(i)
            result = list(zip(obj.num_divisors, obj.sum_divisors))
            self.assertEqual(result, expected[:i+1], i)

    def testing_sieve_of_min_primes_on_powers_10_up_to_1m(self):
        """tested up to 1m"""
        limit = 2*(10 ** 6)
        power_limit = 7
        test_obj = factorizations.MathAlgorithms()
        obj = prime_sieve_variants.MathAlgorithms()
        test_obj.fill_min_primes_list(limit)
        expected = [(1, 1), (1, 1)]
        app = expected.append
        for i in range(2, limit):
            factors = test_obj.prime_factorize_n_log_n(i)
            expected_num = expected_sum = 1
            for prime, power in factors.items():
                expected_num *= (power + 1)
                expected_sum *= ((prime ** (power + 1) - 1) // (prime - 1))
            app((expected_num, expected_sum))
        for i in range(1, power_limit):
            cur = 10 ** i
            obj.num_and_sum_of_divisors_faster(cur)
            result = list(zip(obj.num_divisors, obj.sum_divisors))
            self.assertEqual(result, expected[:cur+1], i)

    def testing_sieve_of_min_primes_on_powers_2_up_to_1m(self):
        """tested up to 1m"""
        limit = 2**21
        power_limit = 21
        test_obj = factorizations.MathAlgorithms()
        obj = prime_sieve_variants.MathAlgorithms()
        test_obj.fill_min_primes_list(limit)
        expected = [(1, 1), (1, 1)]
        app = expected.append
        for i in range(2, limit):
            factors = test_obj.prime_factorize_n_log_n(i)
            expected_num = expected_sum = 1
            for prime, power in factors.items():
                expected_num *= (power + 1)
                expected_sum *= ((prime ** (power + 1) - 1) // (prime - 1))
            app((expected_num, expected_sum))
        for i in range(1, power_limit):
            cur = 2 ** i
            obj.num_and_sum_of_divisors_faster(cur)
            result = list(zip(obj.num_divisors, obj.sum_divisors))
            self.assertEqual(result, expected[:cur+1], i)

    def testing_num_and_sum_of_divisors_same_as_faster_version_2500_runs(self):
        """tested up to 10k"""
        obj1 = prime_sieve_variants.MathAlgorithms()
        obj2 = prime_sieve_variants.MathAlgorithms()
        limit = 2500
        for i in range(2, limit):
            obj1.num_and_sum_of_divisors(limit)
            obj2.num_and_sum_of_divisors_faster(limit)
            self.assertEqual(obj1.num_divisors, obj2.num_divisors)
            self.assertEqual(obj1.sum_divisors, obj2.sum_divisors)

    def testing_num_and_sum_of_divisors_same_as_faster_version_powers_10(self):
        """tested up to 10m before."""
        limit = 8
        obj1 = prime_sieve_variants.MathAlgorithms()
        obj2 = prime_sieve_variants.MathAlgorithms()
        for power in range(1, limit):
            n_limit = 10 ** power
            obj1.num_and_sum_of_divisors(n_limit)
            obj2.num_and_sum_of_divisors_faster(n_limit)
            self.assertListEqual(obj1.num_divisors, obj2.num_divisors)
            self.assertListEqual(obj1.sum_divisors, obj2.sum_divisors)

    def testing_num_and_sum_of_divisors_same_as_faster_version_powers_2(self):
        """tested up to 10m before."""
        limit = 23
        obj1 = prime_sieve_variants.MathAlgorithms()
        obj2 = prime_sieve_variants.MathAlgorithms()
        for power in range(1, limit):
            n_limit = 2 ** power
            obj1.num_and_sum_of_divisors(n_limit)
            obj2.num_and_sum_of_divisors_faster(n_limit)
            self.assertListEqual(obj1.num_divisors, obj2.num_divisors)
            self.assertListEqual(obj1.sum_divisors, obj2.sum_divisors)

    def testing_euler_phi_plus_sum_and_number_of_diff_prime_factors_5k_runs(self):
        """tested up to 10k"""
        limit = 5000
        test_obj = factorizations.MathAlgorithms()
        test_obj.fill_min_primes_list(limit)
        obj = prime_sieve_variants.MathAlgorithms()
        expected_1 = [0, 0]
        expected_2 = [0, 0]
        expected_3 = [0, 1]
        for i in range(2, limit):
            factors = test_obj.prime_factorize_n_log_n(i)
            expected_phi = i
            for prime in factors:
                expected_phi -= (expected_phi // prime)
            expected_1.append(len(factors))
            expected_2.append(sum(factors))
            expected_3.append(expected_phi)
        for i in range(2, limit):
            obj.euler_phi_plus_sum_and_number_of_diff_prime_factors(i)
            self.assertEqual(obj.num_diff_prime_factors, expected_1[:i+1], i)
            self.assertEqual(obj.sum_diff_prime_factors, expected_2[:i+1], i)
            self.assertEqual(obj.euler_phi, expected_3[:i+1], i)

    def testing_euler_phi_plus_sum_and_number_of_diff_prime_factors_powers_10(self):
        """tested up to 10m before."""
        power_limit = 6
        limit = 2*(10 ** power_limit)
        test_obj = factorizations.MathAlgorithms()
        test_obj.fill_min_primes_list(limit)
        obj = prime_sieve_variants.MathAlgorithms()
        expected_1 = [0, 0]
        expected_2 = [0, 0]
        expected_3 = [0, 1]
        app1, app2, app3 = expected_1.append, expected_2.append, expected_3.append
        for i in range(2, limit):
            factors = test_obj.prime_factorize_n_log_n(i)
            expected_phi = i
            for prime in factors:
                expected_phi -= (expected_phi // prime)
            app1(len(factors))
            app2(sum(factors))
            app3(expected_phi)
        for power in range(1, power_limit):
            n_limit = 10 ** power
            obj.euler_phi_plus_sum_and_number_of_diff_prime_factors(n_limit)
            self.assertEqual(obj.num_diff_prime_factors, expected_1[:n_limit+1], n_limit)
            self.assertEqual(obj.sum_diff_prime_factors, expected_2[:n_limit+1], n_limit)
            self.assertEqual(obj.euler_phi, expected_3[:n_limit+1], n_limit)

    def testing_euler_phi_plus_sum_and_number_of_diff_prime_factors_powers_2(self):
        """tested up to 10m before."""
        power_limit = 21
        limit = 2 ** power_limit
        test_obj = factorizations.MathAlgorithms()
        test_obj.fill_min_primes_list(limit)
        obj = prime_sieve_variants.MathAlgorithms()
        expected_1 = [0, 0]
        expected_2 = [0, 0]
        expected_3 = [0, 1]
        app1, app2, app3 = expected_1.append, expected_2.append, expected_3.append
        for i in range(2, limit):
            factors = test_obj.prime_factorize_n_log_n(i)
            expected_phi = i
            for prime in factors:
                expected_phi -= (expected_phi // prime)
            app1(len(factors))
            app2(sum(factors))
            app3(expected_phi)
        for power in range(1, power_limit):
            n_limit = 2 ** power
            obj.euler_phi_plus_sum_and_number_of_diff_prime_factors(n_limit)
            self.assertEqual(obj.num_diff_prime_factors, expected_1[:n_limit+1], n_limit)
            self.assertEqual(obj.sum_diff_prime_factors, expected_2[:n_limit+1], n_limit)
            self.assertEqual(obj.euler_phi, expected_3[:n_limit+1], n_limit)

    def testing_num_and_sum_of_prime_factors_5k_runs(self):
        """tested up to 10k"""
        limit = 5000
        test_obj = factorizations.MathAlgorithms()
        test_obj.fill_min_primes_list(limit)
        obj = prime_sieve_variants.MathAlgorithms()
        expected_1 = [0, 0]
        expected_2 = [0, 0]
        app1, app2 = expected_1.append, expected_2.append
        for i in range(2, limit):
            factors = test_obj.prime_factorize_n_log_n(i)
            app1(sum(factors.values()))
            app2(sum(prime * power for prime, power in factors.items()))
        for i in range(10, limit):
            obj.num_and_sum_of_prime_factors(i)
            self.assertEqual(obj.num_prime_factors, expected_1[:i+1], i)
            self.assertEqual(obj.sum_prime_factors, expected_2[:i+1], i)

    def testing_num_and_sum_of_prime_factors_powers_10(self):
        """tested up to 10m before."""
        power_limit = 6
        limit = 2*(10 ** power_limit)
        test_obj = factorizations.MathAlgorithms()
        test_obj.fill_min_primes_list(limit)
        obj = prime_sieve_variants.MathAlgorithms()
        expected_1 = [0, 0]
        expected_2 = [0, 0]
        app1, app2 = expected_1.append, expected_2.append
        for i in range(2, limit):
            factors = test_obj.prime_factorize_n_log_n(i)
            app1(sum(factors.values()))
            app2(sum(prime * power for prime, power in factors.items()))
        for power in range(1, power_limit):
            n_limit = 10 ** power
            obj.num_and_sum_of_prime_factors(n_limit)
            self.assertEqual(obj.num_prime_factors, expected_1[:n_limit+1], n_limit)
            self.assertEqual(obj.sum_prime_factors, expected_2[:n_limit+1], n_limit)

    def testing_num_and_sum_of_prime_factors_powers_2(self):
        """tested up to 10m before."""
        power_limit = 21
        limit = 2*(2 ** power_limit)
        test_obj = factorizations.MathAlgorithms()
        test_obj.fill_min_primes_list(limit)
        obj = prime_sieve_variants.MathAlgorithms()
        expected_1 = [0, 0]
        expected_2 = [0, 0]
        app1, app2 = expected_1.append, expected_2.append
        for i in range(2, limit):
            factors = test_obj.prime_factorize_n_log_n(i)
            app1(sum(factors.values()))
            app2(sum(prime * power for prime, power in factors.items()))
        for power in range(1, power_limit):
            n_limit = 2 ** power
            obj.num_and_sum_of_prime_factors(n_limit)
            self.assertEqual(obj.num_prime_factors, expected_1[:n_limit+1], n_limit)
            self.assertEqual(obj.sum_prime_factors, expected_2[:n_limit+1], n_limit)

    def test_prime_factorize_n_log_n_against_prime_factorize_n_trivial_1m(self):
        """Assumes prime_factorize_n_trivial works."""
        limit = 1000000
        obj = factorizations.MathAlgorithms()
        obj.fill_min_primes_list(limit)
        for number in range(2, limit):
            expected_factors = obj.prime_factorize_n_trivial(number)
            result_factors = obj.prime_factorize_n_log_n(number)
            self.assertEqual(len(expected_factors), len(result_factors))
            self.assertEqual(result_factors.items(), expected_factors.items())

    def test_prime_factorize_against_prime_factorize_n_log_n_1m(self):
        """Assumes prime_factorize_n_trivial works."""
        limit = 1000000
        obj = factorizations.MathAlgorithms()
        obj.fill_min_primes_list(limit)
        for number in range(2, limit):
            expected_factors = obj.prime_factorize_n_log_n(number)
            result_factors = obj.prime_factorize_n(number)
            self.assertEqual(len(expected_factors), len(result_factors))
            self.assertEqual(result_factors.items(), expected_factors.items())

    def test_prime_factorize_n_variants_1m(self):
        limit = 1000000
        obj_base = prime_sieve_variants.MathAlgorithms()
        obj = factorizations.MathAlgorithms()
        obj_base.num_and_sum_of_divisors_faster(limit)
        obj_base.num_and_sum_of_prime_factors(limit)
        obj_base.euler_phi_plus_sum_and_number_of_diff_prime_factors(limit)
        obj.fill_primes_list_and_set(limit)
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

    def test_binomial_coefficient_n_mod_p_prep_and_c_n_k_up_to_1k(self):
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

    def test_binomial_coefficient_n_mod_p_prep_and_c_n_k_up_to_1k_inverse(self):
        limit = 1000
        mod_m = 10**9+7
        inverse = True
        test_obj = binomial_coefficient.MathAlgorithms()
        obj = binomial_coefficient.MathAlgorithms()
        obj.binomial_coefficient_n_mod_p_prep(limit+1, mod_m, inverse)
        for n in range(limit):
            for k in range(n+1):
                expected = test_obj.binomial_coefficient_dp_with_cache(n, k) % mod_m
                result = obj.c_n_k(n, k, inverse)
                self.assertEqual(result, expected, "{} {}".format(n, k))

    def test_binomial_coefficient_n_mod_p_prep_and_c_n_k_up_to_1m_vs_inverse_random(self):
        limit = 1000000
        mod_m = 10**9+7
        inverse = False
        test_inverse = True
        test_obj = binomial_coefficient.MathAlgorithms()
        obj = binomial_coefficient.MathAlgorithms()
        test_obj.binomial_coefficient_n_mod_p_prep(limit+1, mod_m, test_inverse)
        obj.binomial_coefficient_n_mod_p_prep(limit+1, mod_m, inverse)
        queries = set()
        while len(queries) < limit:
            a, b = randint(0, limit+1), randint(0, limit+1)
            queries.add((a, b) if a < b else (b, a))
        for n, k in queries:
            expected = test_obj.c_n_k(n, k, test_inverse)
            result = obj.c_n_k(n, k, inverse)
            self.assertEqual(result, expected, "{} {}".format(n, k))

    def test_binomial_coefficient_n_mod_p_prep_inv_300(self):
        """tested up to 500 but takes long."""
        limit = 300
        mod_m = 10**9+7
        obj_inv_2 = True
        obj_2 = binomial_coefficient.MathAlgorithms()
        obj_3 = binomial_coefficient.MathAlgorithms()
        for max_n in range(10, limit+1):
            obj_2.mod_p = 0
            obj_2.binomial = {}
            obj_2.fact = []
            obj_2.inv_fact = []
            obj_2.binomial_coefficient_n_mod_p_prep(max_n+1, mod_m, obj_inv_2)
            for n in range(max_n+1):
                for k in range(n+1):
                    expected = obj_3.binomial_coefficient_dp_with_cache(n, k) % mod_m
                    result_2 = obj_2.c_n_k(n, k, obj_inv_2)
                    self.assertEqual(result_2, expected, "{} {} {}".format(max_n, n, k))

    def test_binomial_coefficient_n_mod_p_prep_no_inv_200(self):
        """tested up to 500 but takes long."""
        limit = 200
        mod_m = 10**9+7
        obj_inv_1 = False
        obj_1 = binomial_coefficient.MathAlgorithms()
        obj_3 = binomial_coefficient.MathAlgorithms()
        for max_n in range(10, limit+1):
            obj_1.mod_p = 0
            obj_1.binomial = {}
            obj_1.fact = []
            obj_1.inv_fact = []
            obj_1.binomial_coefficient_n_mod_p_prep(max_n+1, mod_m, obj_inv_1)
            for n in range(max_n+1):
                for k in range(n+1):
                    expected = obj_3.binomial_coefficient_dp_with_cache(n, k) % mod_m
                    result_1 = obj_1.c_n_k(n, k, obj_inv_1)
                    self.assertEqual(result_1, expected, "{} {} {}".format(max_n, n, k))

    def test_fft_prepare_swap_indices_hard_coded_tests(self):
        obj = fast_fourier_transform.MathAlgorithms()
        obj.fft_prepare_swap_indices(16)
        expected = [(1, 8), (2, 4), (3, 12), (5, 10), (7, 14), (11, 13)]
        self.assertEqual(obj.fft_swap_indices, expected)

    def test_fft_prepare_swap_indices_thorough_4m(self):
        """tested up to 16m"""
        limit = 22
        binary_strings = [bin(i)[-1:1:-1] for i in range(2**limit)]
        obj = fast_fourier_transform.MathAlgorithms()
        def get_reversed(n):
            def get_rev(num, bit_size):
                return int(binary_strings[num].ljust(bit_size, '0'), 2)
            bit = n.bit_length()-1
            ans = [(i, get_rev(i, bit)) for i in range(n)]
            return [(a, b) for a, b in ans if a < b]
        for i in range(1, limit):
            expected = get_reversed(2**i)
            obj.fft_prepare_swap_indices(2**i)
            self.assertEqual(obj.fft_swap_indices, expected)

    def test_fft_prepare_lengths_list_32(self):
        expected = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                    65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216,
                    33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
                    4294967296]
        obj = fast_fourier_transform.MathAlgorithms()
        limit = 32
        for i in range(1, limit + 1):
            obj.fft_prepare_lengths_list(2**i)
            self.assertEqual(obj.fft_lengths, expected[:i])

    def test_fft_multiply_in_place_1k(self):
        obj = fast_fourier_transform.MathAlgorithms()
        limit = 1000
        for i in range(1, limit+1):
            for j in range(1, limit+1):
                expected = i*j
                a, b = list(map(int, str(i))), list(map(int, str(j)))
                result = obj.fft_multiply_in_place(a[::-1], b[::-1])
                self.assertEqual(result, list(map(int, str(expected))))

    def test_fft_multiply_in_place_100k_large_random_100_bit_size(self):
        limit_power = 101
        limit = 100000
        queries = [(randint(1, 2**limit_power), randint(1, 2**limit_power)) for _ in range(limit)]
        obj = fast_fourier_transform.MathAlgorithms()
        for a, b in queries:
            expected = a*b
            a_vec, b_vec = list(map(int, str(a))), list(map(int, str(b)))
            result = obj.fft_multiply_in_place(a_vec[::-1], b_vec[::-1])
            self.assertEqual(result, list(map(int, str(expected))))

    def test_extended_euclid_recursive_vs_extended_euclid_iterative_1000(self):
        """tested on 10k but takes a while TODO test some base cases"""
        limit = 1000
        obj_1 = chinese_remainder_theorem.MathAlgorithms()
        obj_2 = chinese_remainder_theorem.MathAlgorithms()
        for a in range(1, limit + 1):
            for b in range(1, limit + 1):
                result_1 = obj_1.extended_euclid_recursive(a, b)
                result_2 = obj_2.extended_euclid_iterative(a, b)
                self.assertEqual(result_1, result_2, "{} {}".format(a, b))

    def test_extended_euclid_recursive_vs_extended_euclid_iterative_1m_random_100_bit_size(self):
        """tested on 10k but takes a while TODO test some base cases"""
        limit = 1000000
        limit_power = 100
        obj_1 = chinese_remainder_theorem.MathAlgorithms()
        obj_2 = chinese_remainder_theorem.MathAlgorithms()
        queries = [(2**randint(10, limit_power), 2**randint(10, limit_power))
                   for _ in range(limit)]
        for a, b in queries:
            result_1 = obj_1.extended_euclid_recursive(a, b)
            result_2 = obj_2.extended_euclid_iterative(a, b)
            self.assertEqual(result_1, result_2, "{} {}".format(a, b))
