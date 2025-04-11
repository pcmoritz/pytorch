import torch
from torch import nn
import unittest

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TestTT(unittest.TestCase):

    def test_tt_roundtrip(self):
        for dtype in [torch.bfloat16, torch.float32]:
            a = torch.ones(100000, dtype=dtype)
            b = a.to("tt")
            c = b.to("cpu")
            self.assertTrue((a == c).all())

    def test_tt_add(self):
        a = torch.ones(32 * 32, dtype=torch.bfloat16)
        b = a.to("tt")
        c = b + b
        d = c.to("cpu")
        self.assertTrue((d == 2.0).all())

    def test_tt_relu(self):
        a = torch.ones(32 * 32, dtype=torch.bfloat16)
        b = torch.relu(a)
        c = torch.relu(a.to("tt"))
        self.assertTrue(torch.allclose(b, c.to("cpu"), rtol=1e-2))

    def test_tt_mm(self):
        a = torch.rand(64, 128, dtype=torch.bfloat16) - 0.5
        b = torch.rand(128, 96, dtype=torch.bfloat16) - 0.5
        c = a @ b
        d = (a.to("tt") @ b.to("tt")).to("cpu")
        print("n", torch.linalg.norm(c - d))
        self.assertTrue(torch.allclose(c, d.to("cpu"), rtol=1e-2, atol=1e-1))

    def test_tt_addmm(self):
        M = torch.randn(64, 128, dtype=torch.bfloat16)
        mat1 = torch.randn(64, 32, dtype=torch.bfloat16)
        mat2 = torch.randn(32, 128, dtype=torch.bfloat16)
        a = torch.addmm(M, mat1, mat2)
        b = torch.addmm(M.to("tt"), mat1.to("tt"), mat2.to("tt")).to("cpu")
        self.assertTrue(torch.allclose(a, b, rtol=1e-2, atol=1e-1))

    def test_tt_network(self):
        model = NeuralNetwork().to("tt")
        X = torch.rand(1, 28, 28, dtype=torch.bfloat16, device="tt")
        print(X.cpu())
        logits = model.bfloat16().to("tt")(X)
        print(logits.cpu())

if __name__ == "__main__":
    unittest.main()

