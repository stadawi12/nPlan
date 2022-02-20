import sys
sys.path.insert(1, "../lib/models")
import unittest
import torch

# importing a model from ../lib/models
import linear


class TestModels(unittest.TestCase):


    def test_output_shape(self):
        import linear
        model = linear.Linear()
        # generate a random input sample
        inpt = torch.randn(2, 50)
        out  = model(inpt)

        self.assertEqual(out.shape, torch.Size([2, 121]))

    def test_output_range(self):
        import linear
        model = linear.Linear()
        # generate a random input sample and ensure the output is
        # between the range 0 and 1
        inpt = torch.randn(100, 50)
        out  = model(inpt)
        self.assertTrue(out.max() < 1)
        self.assertTrue(out.min() > 0)

    def test_output_shape_conv(self):
        import convNet
        model = convNet.UNet()
        # generate a random input sample
        inpt = torch.randn(2, 50)
        out  = model(inpt)

        self.assertEqual(out.shape, torch.Size([2, 121]))

    def test_output_range_conv(self):
        import convNet
        model = convNet.UNet()
        # generate a random input sample and ensure the output is
        # between the range 0 and 1
        inpt = torch.randn(2, 50)
        out  = model(inpt)
        self.assertTrue(out.max() < 1)
        self.assertTrue(out.min() > 0)


if __name__ == '__main__':
    unittest.main()
