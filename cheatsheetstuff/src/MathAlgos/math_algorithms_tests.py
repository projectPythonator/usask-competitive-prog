import unittest
from random import randint
import primality_testing as prime_tests


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
            number = randint(2**50, 2**64)
            self.assertEqual(test_obj.is_prime_trivial(number),
                             obj.miller_rabin_primality_test(number), number)
