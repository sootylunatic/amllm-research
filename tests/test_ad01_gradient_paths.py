"""AD01 gradient audit: where does a training gradient flow when tokens exit early? (autograd, no approximation)

Facts established here, each by a test:
  1. hard forward: a token that has exited receives gradient only from the hops it ran; later hops get exactly zero from
     its loss. The threshold decision is not differentiable, but the survival WEIGHTS of the hops that did run are: a token
     that ran two hops gives the halt head a (non-zero) LM gradient through the weight of its second hop, a token that ran
     only one hop (weight 1) gives none;
  2. soft forward (the training path): the halt head DOES get gradient, through the survival weights, and later hops get a
     gradient from every token, scaled by its survival;
  3. the ponder penalty is the only direct gradient on the halt head that does not depend on the language-model loss;
  4. exit loss (AD02a): the loss of the state after hop h reaches the parameters of hops <= h and the shared head, and
     exactly zero for hops > h.
"""
import pytest
import torch
from torch.nn import functional as F

from amllm.model import build_model
from test_model import tiny_cfg


def make(**kw):
    torch.manual_seed(5)
    cfg = tiny_cfg(n_hops=3, unshared_attention=True, route_weighting="full_softmax", **kw)
    return build_model(cfg)


def hop_param_grad_norms(model):
    """Gradient norm per hop of the hop-specific parameters (attention of hop h; the bank is shared by all hops)."""
    out = {}
    for h in range(model.cfg.n_hops):
        ps = list(model.attention.parameters()) if h == 0 else list(model.hop_attentions[h - 1].parameters())
        out[h] = sum(float(p.grad.norm() ** 2) for p in ps if p.grad is not None) ** 0.5
    return out


def test_hard_forward_gives_zero_gradient_to_hops_after_the_exit_and_none_to_the_halt_head_for_a_one_hop_exit():
    model = make().eval()
    x = torch.randint(0, 256, (1, 6))
    y = torch.randint(0, 256, (1, 6))
    with torch.no_grad():
        model.halt_head.weight.zero_()
        model.halt_head.bias.fill_(8.0)   # p_stop ~ 1 after hop 1: every token runs exactly one hop in the hard path
    logits, info = model(x, hard_halt=True)
    assert float(info["hops_per_token"].float().max()) == 1.0
    F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).backward()
    norms = hop_param_grad_norms(model)
    assert norms[0] > 0
    assert norms[1] == 0.0 and norms[2] == 0.0              # hops after the exit never saw the tokens' loss
    hg = model.halt_head.bias.grad
    assert hg is None or float(hg.abs().sum()) == 0.0       # a one-hop exit runs with weight 1: nothing reaches the head


def test_hard_forward_two_hop_exit_reaches_the_halt_head_through_the_survival_weight():
    model = make().eval()
    x = torch.randint(0, 256, (1, 6))
    y = torch.randint(0, 256, (1, 6))
    with torch.no_grad():
        model.halt_head.weight.zero_()
        model.halt_head.bias.fill_(-0.8473)   # p_stop = 0.3: survival 0.7 after hop 1 (continues), 0.49 after hop 2 (stops)
    logits, info = model(x, hard_halt=True)
    assert float(info["hops_per_token"].float().min()) == 2.0 == float(info["hops_per_token"].float().max())
    F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).backward()
    norms = hop_param_grad_norms(model)
    assert norms[0] > 0 and norms[1] > 0 and norms[2] == 0.0  # the third hop never ran for these tokens
    assert float(model.halt_head.bias.grad.abs().sum()) > 0   # LM gradient through the survival weight of hop 2


def test_soft_forward_gives_the_halt_head_a_gradient_and_later_hops_a_scaled_one():
    model = make().train()
    x, y = torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        model.halt_head.weight.zero_()
        model.halt_head.bias.fill_(2.0)   # survival decays fast, but is nowhere exactly zero
    logits, info = model(x, hard_halt=False)
    F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).backward()   # the language-model loss ONLY (no ponder)
    assert float(model.halt_head.bias.grad.abs()) > 0         # soft training does teach the stopping decision ...
    norms = hop_param_grad_norms(model)
    assert norms[1] > 0 and norms[2] > 0                      # ... and later hops get gradient from every token


def test_ponder_penalty_alone_pushes_the_halt_head_toward_stopping():
    model = make(ponder_coefficient=1.0).train()
    x = torch.randint(0, 256, (2, 8))
    _, info = model(x, hard_halt=False)
    (info["aux_loss"]).backward()                              # ponder + balance, no LM loss
    g = model.halt_head.bias.grad
    assert g is not None and float(g) < 0     # d(ponder)/d(bias) < 0: gradient descent RAISES the bias, i.e. stops earlier


def test_exit_loss_reaches_only_the_hops_up_to_the_exit_and_the_shared_head():
    model = make(halt_rule="none", anytime_weight=1.0).train()
    x, y = torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8))
    _, info = model(x, hard_halt=False)
    states = info["hop_states"]
    F.cross_entropy(model._head(states[0]).reshape(-1, 256), y.reshape(-1)).backward()   # exit loss after hop 1 only
    norms = hop_param_grad_norms(model)
    assert norms[0] > 0 and norms[1] == 0.0 and norms[2] == 0.0
    assert model.lm_head.weight.grad is not None and float(model.lm_head.weight.grad.abs().sum()) > 0   # shared head trained
    assert float(model.bank.experts.w1.grad.abs().sum()) > 0   # the shared bank is trained by the hop-1 exit as well
