from dataclasses import asdict, dataclass, fields

ARCHS = ("amllm", "dense", "moe")
HALT_RULES = ("survival", "step", "none")


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "amllm"
    vocab_size: int = 256   # Byte tokens: demonstrator, not a real LLM tokenizer
    max_seq_len: int = 128
    d_model: int = 64
    n_heads: int = 4
    # dense / moe controls: a fixed stack of n_layers blocks, each with its own attention
    n_layers: int = 4
    ffn_hidden: int = 256     # dense FFN width
    n_experts: int = 8        # moe experts per layer
    moe_topk: int = 2
    # amllm: one shared module bank applied for up to n_hops recurrent hops
    n_modules: int = 8
    experts_per_module: int = 4
    module_topk: int = 2
    expert_topk: int = 2
    expert_hidden: int = 128  # hidden width of one expert FFN (amllm and moe)
    n_hops: int = 4
    attn_per_hop: bool = True  # False = v0.1 topology: a single attention before all hops
    unshared_attention: bool = False  # True = each hop has its own attention weights (bank stays shared)
    # survival: continue while cumulative survival >= threshold, updates weighted by survival
    #           (hard inference is soft training truncated where survival is small)
    # step:     v0.1 rule, stop when the per-hop stop probability >= threshold, unweighted
    # none:     fixed depth, no halting head in use
    halt_rule: str = "survival"
    halt_threshold: float = 0.5
    halt_bias_init: float = -1.0
    # route weights: topk_softmax (v0.1/v0.2 default; k=1 gives the router no LM gradient) or
    # full_softmax (Switch-style, gradient for every k) or straight_through (forward of topk_softmax, gradient for
    # every k); router_mode=random is a no-learned-routing control
    route_weighting: str = "topk_softmax"
    router_mode: str = "learned"
    ponder_coefficient: float = 0.002
    balance_coefficient: float = 0.01
    # FD01: number of distinct module banks; hop h uses bank h * bank_groups // n_hops (1 = one bank shared by all hops,
    # n_hops = a separate bank per hop). Banks of groups > 0 are created after every other module, so defaults are unchanged.
    bank_groups: int = 1
    # FD03: normalisation in front of the bank. shared = one LayerNorm for all hops; per_hop = one LayerNorm per hop;
    # hop_affine = the shared LayerNorm followed by a per-hop scale and shift (initialised to identity)
    bank_norm: str = "shared"
    # AD02a: weight of the anytime (multi-exit) loss. 0 = off. > 0: in training the state after every hop except the last
    # is read out through the final norm and the shared head and trained on the next token; loss += weight * mean over hops
    anytime_weight: float = 0.0

    def __post_init__(self):
        assert self.arch in ARCHS, self.arch
        assert self.halt_rule in HALT_RULES, self.halt_rule
        assert self.route_weighting in ("topk_softmax", "full_softmax", "straight_through"), self.route_weighting
        assert self.router_mode in ("learned", "random"), self.router_mode
        assert self.anytime_weight >= 0.0, self.anytime_weight
        assert self.bank_norm in ("shared", "per_hop", "hop_affine"), self.bank_norm
        assert 1 <= self.bank_groups <= self.n_hops, (self.bank_groups, self.n_hops)
        assert self.d_model % self.n_heads == 0
        assert 1 <= self.module_topk <= self.n_modules
        assert 1 <= self.expert_topk <= self.experts_per_module
        assert 1 <= self.moe_topk <= self.n_experts
        assert self.n_hops >= 1 and self.n_layers >= 1
        assert self.max_seq_len > 1
        assert 0 < self.halt_threshold < 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
