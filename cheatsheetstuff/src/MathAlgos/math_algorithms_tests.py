import unittest
import primality_testing as prime_tests


class TestMathMethods(unittest.TestCase):
    def test_is_prime_trivial_up_to_100(self):
        primes_to_100 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                         43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        obj = prime_tests.MathAlgorithms()
        for number in range(100):
            if obj.is_prime_trivial(number):
                self.assertTrue(number in primes_to_100)
            else:
                self.assertTrue(number not in primes_to_100)

    def test_is_prime_trivial_vs_sieve_up_to_n(self):
        limit = 1000000
        obj = prime_tests.MathAlgorithms()
        obj.sieve_of_eratosthenes(limit)
        primes_generated_by_sieve = set(obj.primes_list)
        for number in range(limit):
            if obj.is_prime_trivial(number):
                self.assertTrue(number in primes_generated_by_sieve)
            else:
                self.assertTrue(number not in primes_generated_by_sieve)
