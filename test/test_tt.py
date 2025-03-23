import torch
import unittest
from parameterized import parameterized

class TestTTDevice(unittest.TestCase):
    @parameterized.expand([
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
    ])
    def test_to_tt_and_back(self, name, dtype):
        a = torch.ones(100000, dtype=dtype)
        b = a.to("tt")
        c = b.to("cpu")
        self.assertTrue((a == c).all())

if __name__ == "__main__":
    unittest.main()