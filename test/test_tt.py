import torch
from torch import nn
import unittest

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 32, 512, bias=False, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(512, 32, bias=False, dtype=torch.bfloat16),
        )

    def forward(self, x):
        with torch.no_grad():
            logits = self.linear_relu_stack(x)
        return logits


class TestTT(unittest.TestCase):

    def test_tt_roundtrip(self):
        for dtype in [torch.bfloat16, torch.float32, torch.long]:
            a = torch.ones(100000, dtype=dtype)
            b = a.to("tt")
            c = b.to("cpu")
            self.assertTrue((a == c).all())

    def test_tt_add(self):
        a = torch.ones(64 * 32, dtype=torch.bfloat16)
        b = a.to("tt")
        c = b + b
        d = c.to("cpu")
        self.assertTrue((d == 2.0).all())

    def test_tt_mul(self):
        a = 2.0 * torch.ones(64 * 32, dtype=torch.bfloat16)
        b = a.to("tt")
        c = b * b
        d = c.to("cpu")
        self.assertTrue((d == 4.0).all())

    def test_tt_eltwise(self):
        for func in [torch.relu, torch.cos, torch.sin]:
            a = torch.rand(32 * 32, dtype=torch.bfloat16) - 0.5
            b = func(a)
            c = func(a.to("tt"))
            self.assertTrue(torch.allclose(b, c.to("cpu"), rtol=1e-2))

    def test_tt_pow(self):
        a = torch.rand(32 * 32, dtype=torch.bfloat16) - 0.5
        b = a.pow(2)
        c = a.to("tt").pow(2)
        self.assertTrue(torch.allclose(b, c.to("cpu"), rtol=1e-2))

    def test_tt_mm(self):
        # First test with non-transposed matrices
        a = torch.rand(64, 128, dtype=torch.bfloat16) - 0.5
        b = torch.rand(128, 96, dtype=torch.bfloat16) - 0.5
        c = a @ b
        d = (a.to("tt") @ b.to("tt")).to("cpu")
        self.assertTrue(torch.allclose(c, d.to("cpu"), rtol=1e-2, atol=1e-1))
        # Test with transposed matrix
        b = torch.rand(96, 128, dtype=torch.bfloat16) - 0.5
        c = a @ torch.t(b)
        b = b.to("tt")
        d = (a.to("tt") @ torch.t(b)).to("cpu")
        self.assertTrue(torch.allclose(c, d.to("cpu"), rtol=1e-2, atol=1e-1))

    def test_tt_addmm(self):
        M = torch.randn(64, 128, dtype=torch.bfloat16)
        mat1 = torch.randn(64, 32, dtype=torch.bfloat16)
        mat2 = torch.randn(32, 128, dtype=torch.bfloat16)
        a = torch.addmm(M, mat1, mat2)
        b = torch.addmm(M.to("tt"), mat1.to("tt"), mat2.to("tt")).to("cpu")
        self.assertTrue(torch.allclose(a, b, rtol=1e-2, atol=1e-1))

    def test_tt_network(self):
        model = NeuralNetwork()
        X = torch.rand(64, 32 * 32, dtype=torch.bfloat16) - 0.5
        logits1 = model(X)
        logits2 = model.to("tt")(X.to("tt")).to("cpu")
        self.assertTrue(torch.allclose(logits1, logits2, rtol=1e-2, atol=1e-1))

    def test_tt_embedding(self):
        vocab_size = 64
        n_embed = 32
        embed = nn.Embedding(vocab_size, n_embed).to(torch.bfloat16)
        indices = torch.randint(vocab_size, (128,), dtype=torch.int32)
        result_cpu = embed.forward(indices)
        result_tt = embed.to("tt").forward(indices.to("tt")).cpu()
        self.assertTrue(torch.allclose(result_cpu, result_tt))

    def test_tt_cat(self):
        a = torch.rand(1, 2, 32, dtype=torch.bfloat16)
        b = torch.rand(1, 3, 32, dtype=torch.bfloat16)
        c = torch.cat([a, b], dim=1)
        c_tt = torch.cat([a.to("tt"), b.to("tt")], dim=1).to("cpu")
        self.assertTrue(torch.allclose(c, c_tt))

    def test_tt_mean(self):
        a = torch.rand(1, 1, 32, 32, dtype=torch.bfloat16) - 0.5
        b = a.mean(dim=-1)
        b_tt = a.to("tt").mean(dim=-1).to("cpu")
        print("b", b)
        print("b_tt", b_tt)
        self.assertTrue(torch.allclose(b, b_tt, rtol=1e-2, atol=1e-1))

if __name__ == "__main__":
    unittest.main()

