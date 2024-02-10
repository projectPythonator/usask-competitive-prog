import unittest
from random import randint
import primality_testing as prime_tests
import prime_sieves
import prime_sieve_variants
import factorizations


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
        test_limit = 10000000
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

    def test_sieve_of_eratosthenes_optimized_10m(self):
        limit = 10000000
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

    def testing_num_and_sum_of_divisors_same_as_faster_version_10m(self):
        """Depends on sieve_of_eratosthenes_optimized and prime_factorize_n working."""
        limit = 10000000
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
