import sys
sys.path.insert(1, "../lib/models")
import unittest
import linear


class TestModels(unittest.TestCase):

    model = linear.Linear()

    def test_output_shape(self):
        pass

if __name__ == '__main__':
    unittest.main()
