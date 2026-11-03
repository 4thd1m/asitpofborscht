"""The op surface of makemore's MLP -- notebooks/3.1_ffn_{torch,borscht}.ipynb --
checked against torch.

Every case is written once, against the API torch and borscht share, and run
twice: on torch tensors, and on borscht copies of the same values.

    TestForward       each expression of the notebook's forward pass and loss
    TestTrainingStep  the notebook's training loop: loss, every `layer.out` and
                      its retained grad, every param's grad, the SGD update and
                      the BatchNorm running stats
    TestSampling      seeding, randint and multinomial -- borscht's RNG is its
                      own (see test_creation.TestRandn), so these check what the
                      notebook relies on rather than torch's bits

Run from `borscht/`, against a `maturin develop`-ed borscht:

    python -m unittest python/tests/test_ffn.py
"""

import importlib
import operator
import unittest
from types import SimpleNamespace

import borscht
import torch

B, T, V, E = 32, 3, 27, 10  # the notebook's batch, context, vocab and embedding sizes
D = 64                      # its model dimension is 200; scaled down to keep a naive borscht fast
EPS, MOMENTUM, LR = 1e-5, 0.1, 0.01


def F(lib):
    """`import <lib>.nn.functional as F`, as the notebook does."""
    return importlib.import_module(lib.__name__ + ".nn.functional")


def to_borscht(t):
    return borscht.tensor(t.tolist())


def assert_matches(actual, expected):
    """borscht `actual` has torch `expected`'s shape and values."""
    torch.testing.assert_close(
        torch.tensor(actual.tolist()), expected.detach(), check_dtype=False
    )


# --------------------------------------------------------------------- forward


def inputs():
    """A tensor at the notebook's shapes for every operand in FORWARD."""
    g = torch.Generator().manual_seed(1337)
    h = 3 * torch.randn(B, D, generator=g)  # pre-activations, wide enough to saturate tanh
    return dict(
        X=torch.randint(0, V, (B, T), generator=g),
        Y=torch.randint(0, V, (B,), generator=g),
        ix=torch.randint(0, B, (B,), generator=g),
        C=torch.randn(V, E, generator=g),
        emb=torch.randn(B, T, E, generator=g),
        x=torch.randn(B, T * E, generator=g),
        W=torch.randn(T * E, D, generator=g),
        b=torch.randn(D, generator=g),
        h=h,
        mean=h.mean(0, keepdim=True),
        var=h.var(0, keepdim=True),
        gamma=torch.randn(D, generator=g),
        beta=torch.randn(D, generator=g),
        running=torch.randn(D, generator=g),
        logits=torch.randn(B, V, generator=g),
    )


# One row per expression in the notebook's forward pass; `t` holds the operands.
FORWARD = {
    # dataloader and embedding
    "Xtr[ix]":          lambda lib, t: t.X[t.ix],
    "C[X]":             lambda lib, t: t.C[t.X],
    "view(-1, T*E)":    lambda lib, t: t.emb.view(-1, T * E),
    "view(B, -1)":      lambda lib, t: t.emb.view(t.emb.shape[0], -1),
    # Linear
    "kaiming init":     lambda lib, t: t.W * (5 / 3) / (T * E) ** 0.5,
    "x @ W":            lambda lib, t: t.x @ t.W,
    "x @ W += b":       lambda lib, t: operator.iadd(t.x @ t.W, t.b),
    # BatchNorm1D
    "mean(0, keepdim)": lambda lib, t: t.h.mean(0, keepdim=True),
    "var(0, keepdim)":  lambda lib, t: t.h.var(0, keepdim=True),  # unbiased, as torch's default
    "normalize":        lambda lib, t: (t.h - t.mean) / lib.sqrt(t.var + EPS),
    "scale and shift":  lambda lib, t: t.gamma * t.h + t.beta,
    "running average":  lambda lib, t: (1 - MOMENTUM) * t.running + MOMENTUM * t.mean,
    # Tanh
    "tanh":             lambda lib, t: lib.tanh(t.h),
    # loss, logging, inference
    "cross_entropy":    lambda lib, t: F(lib).cross_entropy(t.logits, t.Y),
    "log10":            lambda lib, t: t.var.log10(),
    "softmax(dim=1)":   lambda lib, t: F(lib).softmax(t.logits, dim=1),
}


class TestForward(unittest.TestCase):
    def test_matches_torch(self):
        t = inputs()
        b = SimpleNamespace(**{k: to_borscht(v) for k, v in t.items()})
        t = SimpleNamespace(**t)
        for name, f in FORWARD.items():
            with self.subTest(name):
                assert_matches(f(borscht, b), f(torch, t))


# --------------------------------------------------------------- training step

PARAMS = ["C", "W1", "gamma1", "beta1", "W2", "gamma2", "beta2"]
BUFFERS = ["running_mean1", "running_var1", "running_mean2", "running_var2"]


def init():
    """The notebook's model at init, scaled down to one hidden layer --
    Linear -> BatchNorm1D -> Tanh -> Linear -> BatchNorm1D -- and one batch."""
    g = torch.Generator().manual_seed(1337)
    return dict(
        C=torch.randn(V, E, generator=g),
        W1=torch.randn(T * E, D, generator=g) * (5 / 3) / (T * E) ** 0.5,
        gamma1=torch.ones(D), beta1=torch.zeros(D),
        running_mean1=torch.zeros(D), running_var1=torch.ones(D),
        W2=torch.randn(D, V, generator=g) * (5 / 3) / D ** 0.5,
        gamma2=torch.ones(V), beta2=torch.zeros(V),
        running_mean2=torch.zeros(V), running_var2=torch.ones(V),
        X=torch.randint(0, V, (B, T), generator=g),
        Y=torch.randint(0, V, (B,), generator=g),
    )


def train_step(lib, m):
    """One iteration of the notebook's training loop on `m`.  Updates `m` in
    place, as the notebook does, and returns the loss and each layer's out."""

    def batchnorm(x, i):
        xmean, xvar = x.mean(0, keepdim=True), x.var(0, keepdim=True)
        with lib.no_grad():
            m[f"running_mean{i}"] = (1 - MOMENTUM) * m[f"running_mean{i}"] + MOMENTUM * xmean
            m[f"running_var{i}"] = (1 - MOMENTUM) * m[f"running_var{i}"] + MOMENTUM * xvar
        return m[f"gamma{i}"] * ((x - xmean) / lib.sqrt(xvar + EPS)) + m[f"beta{i}"]

    out = {}
    out["linear1"] = m["C"][m["X"]].view(-1, T * E) @ m["W1"]
    out["batchnorm1"] = batchnorm(out["linear1"], 1)
    out["tanh"] = lib.tanh(out["batchnorm1"])
    out["linear2"] = out["tanh"] @ m["W2"]
    out["batchnorm2"] = batchnorm(out["linear2"], 2)
    loss = F(lib).cross_entropy(out["batchnorm2"], m["Y"])

    for o in out.values():
        o.retain_grad()
    for p in PARAMS:
        m[p].grad = None
    loss.backward()
    for p in PARAMS:
        m[p].data += -LR * m[p].grad
    return loss, out


class TestTrainingStep(unittest.TestCase):
    """Two steps from the same init -- two, so that `p.grad = None` and the SGD
    update both have something to get wrong."""

    def test_matches_torch(self):
        mt = init()
        mb = {k: to_borscht(v) for k, v in mt.items()}
        for m in (mt, mb):
            for p in PARAMS:
                m[p].requires_grad = True

        for step in range(2):
            loss_t, out_t = train_step(torch, mt)
            loss_b, out_b = train_step(borscht, mb)
            pairs = {"loss": (loss_b, loss_t)}
            pairs |= {f"{k}.out": (out_b[k], out_t[k]) for k in out_t}
            pairs |= {f"{k}.out.grad": (out_b[k].grad, out_t[k].grad) for k in out_t}
            pairs |= {f"{p}.grad": (mb[p].grad, mt[p].grad) for p in PARAMS}
            pairs |= {k: (mb[k], mt[k]) for k in PARAMS + BUFFERS}
            for name, (actual, expected) in pairs.items():
                with self.subTest(step=step, tensor=name):
                    assert_matches(actual, expected)
            with self.subTest(step=step, tensor="loss.log10().item()"):
                self.assertAlmostEqual(loss_b.log10().item(), loss_t.log10().item(), places=5)


# -------------------------------------------------------------------- sampling


class TestSampling(unittest.TestCase):
    def test_manual_seed_reproduces_draws(self):
        def draw():
            g = borscht.Generator().manual_seed(1337)
            return borscht.randn((V, E), generator=g).tolist()

        self.assertEqual(draw(), draw())

    def test_randint(self):
        ix = borscht.randint(0, 50, (B,))
        self.assertEqual(tuple(ix.shape), tuple(torch.randint(0, 50, (B,)).shape))
        self.assertTrue(all(isinstance(i, int) and 0 <= i < 50 for i in ix.tolist()))

    def test_multinomial(self):
        # one-hot probs have exactly one possible draw, so torch's is borscht's too
        probs = [[0.0] * 13 + [1.0] + [0.0] * 13]
        g = borscht.Generator().manual_seed(1337)
        draw = borscht.multinomial(borscht.tensor(probs), num_samples=1, replacement=True, generator=g)
        want = torch.multinomial(torch.tensor(probs), num_samples=1, replacement=True)
        self.assertEqual(draw.tolist(), want.tolist())
        self.assertEqual(draw.item(), 13)


if __name__ == "__main__":
    unittest.main()
