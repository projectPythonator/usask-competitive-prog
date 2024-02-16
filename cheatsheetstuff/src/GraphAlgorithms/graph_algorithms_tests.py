import unittest
from random import randint
from gc import collect


class TestGraphMethods(unittest.TestCase):

  @classmethod
  def tearDownClass(cls):
    collect()

