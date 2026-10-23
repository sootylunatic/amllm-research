"""Apply the preregistered W1 FD decision rule to paired per-seed differences (variant - reference, nat/token).

    python tools/decide_w1.py -0.021 -0.015 -0.019
"""
import statistics
import sys

T975_DF2 = 4.303  # two-sided 95% t quantile for n = 3 pairs


def decide(deltas: list[float], threshold: float = 0.010) -> str:
    n = len(deltas)
    assert n == 3, "the W1 rule is defined for three paired seeds"
    mean, sd = statistics.fmean(deltas), statistics.stdev(deltas)
    half = T975_DF2 * sd / n ** 0.5
    lo, hi = mean - half, mean + half
    if mean <= -threshold and hi < 0 and all(d < 0 for d in deltas):
        return "SUPPORTED"
    if mean >= threshold and lo > 0 and all(d > 0 for d in deltas):
        return "NOT_SUPPORTED_IN_SCOPE"
    return "INCONCLUSIVE"


if __name__ == "__main__":
    print(decide([float(x) for x in sys.argv[1:]]))
