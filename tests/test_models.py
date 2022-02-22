import sys
sys.path.insert(1, "../lib/models")
sys.path.insert(1, "../lib")
from loadModel import LoadModel
import unittest
import torch

class TestModels(unittest.TestCase):


    def test_output_shape_Linear(self):
        model = LoadModel('Linear')
        # generate a random input sample
        inpt = torch.randn(2, 50)
        out  = model(inpt)

        self.assertEqual(out.shape, torch.Size([2, 121]))

    def test_output_range_Linear(self):
        model = LoadModel('Linear')
        # generate a random input sample and ensure the output is
        # between the range 0 and 1
        inpt = torch.randn(100, 50)
        out  = model(inpt)
        self.assertTrue(out.max() < 1)
        self.assertTrue(out.min() > 0)

    def test_output_shape_smallLinear(self):
        model = LoadModel('smallLinear')
        # generate a random input sample
        inpt = torch.randn(2, 50)
        out  = model(inpt)

        self.assertEqual(out.shape, torch.Size([2, 121]))

    def test_output_range_smallLinear(self):
        model = LoadModel('smallLinear')
        # generate a random input sample and ensure the output is
        # between the range 0 and 1
        inpt = torch.randn(100, 50)
        out  = model(inpt)
        self.assertTrue(out.max() < 1)
        self.assertTrue(out.min() > 0)

    def test_output_shape_linRes(self):
        model = LoadModel('linRes')
        # generate a random input sample
        inpt = torch.randn(2, 50)
        out  = model(inpt)

        self.assertEqual(out.shape, torch.Size([2, 121]))

    def test_output_range_linRes(self):
        model = LoadModel('linRes')
        # generate a random input sample and ensure the output is
        # between the range 0 and 1
        inpt = torch.randn(100, 50)
        out  = model(inpt)
        self.assertTrue(out.max() < 1)
        self.assertTrue(out.min() > 0)

    def test_output_shape_linResBN(self):
        model = LoadModel('linResBN', batch_norm=True)
        # generate a random input sample
        inpt = torch.randn(2, 50)
        out  = model(inpt)

        self.assertEqual(out.shape, torch.Size([2, 121]))

    def test_output_range_linResBN(self):
        model = LoadModel('linResBN', batch_norm=True)
        # generate a random input sample and ensure the output is
        # between the range 0 and 1
        inpt = torch.randn(100, 50)
        out  = model(inpt)
        self.assertTrue(out.max() < 1)
        self.assertTrue(out.min() > 0)



if __name__ == '__main__':
    unittest.main()
