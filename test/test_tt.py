import torch
import unittest

class TestTT(unittest.TestCase):

    def test_tt_roundtrip(self):
        for dtype in [torch.bfloat16, torch.float32]:
            a = torch.ones(100000, dtype=dtype)
            b = a.to("tt")
            c = b.to("cpu")
            self.assertTrue((a == c).all())

    def test_tt_add(self):
        a = torch.ones(100000, dtype=torch.bfloat16)
        b = a.to("tt")
        c = b + b
        d = c.to("cpu")
        self.assertTrue((d == 2.0).all())

    def test_tt_mm(self):
        a = torch.rand(32, 32, dtype=torch.bfloat16) - 0.5
        b = torch.rand(32, 32, dtype=torch.bfloat16) - 0.5
        c = a @ b
        d = (a.to("tt") @ b.to("tt")).to("cpu")
        print(c - d)
        # self.assertTrue(torch.allclose(c, d.to("cpu")))

if __name__ == "__main__":
    unittest.main()

