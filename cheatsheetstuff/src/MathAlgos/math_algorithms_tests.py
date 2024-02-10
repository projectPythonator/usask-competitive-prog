import unittest
import primality_testing as prime_tests


class TestMathMethods(unittest.TestCase):
    def test_is_prime_trivial(self):
        primes_to_100 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                         43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
        obj = prime_tests.MathAlgorithms()
        for number in range(100):
            if obj.is_prime_trivial(number):
                self.assertTrue(number in primes_to_100)
            else:
                self.assertTrue(number not in primes_to_100)
