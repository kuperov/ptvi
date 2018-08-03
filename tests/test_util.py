import unittest
import torch


class TorchTestCase(unittest.TestCase):
    def assertClose(self, a, b, **kwargs):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        self.assertTrue(torch.allclose(a, b, **kwargs), msg="{} != {}".format(a, b))
