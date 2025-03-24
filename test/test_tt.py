import torch
import unittest

class TestTT(unittest.TestCase):

    def test_tt_roundtrip(self):
        for dtype in [torch.bfloat16, torch.float32]:
            a = torch.ones(100000, dtype=dtype)
            b = a.to("tt")
            c = b.to("cpu")
            self.assertTrue((a == c).all())

if __name__ == "__main__":
    unittest.main()

