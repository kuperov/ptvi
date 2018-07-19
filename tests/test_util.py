import unittest
import torch


class TorchTestCase(unittest.TestCase):

    def assertClose(self, *args, **kwargs):
        self.assertTrue(torch.allclose(*args, **kwargs), msg='{} != {}'.format(
            args[0], args[1]))
