

"""Week w01 answers — reference solution

Implements q01()..q12 for Week 01.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm


def q01() -> float:
    # P(at least one head in 3 flips) = 1 - P(all tails)
    return 1.0 - (0.5 ** 3)


def q02() -> int:
    # number of combinations choosing 3 from 10
    return math.comb(10, 3)


def q03() -> float:
    # P(Python | R) = P(Python ∩ R) / P(R) = 10/18
    return 10 / 18


def q04() -> float:
    # Bayes: P(D|+) = P(+|D)P(D) / [P(+|D)P(D) + P(+|~D)P(~D)]
    pD = 0.01
    p_pos_given_D = 0.95
    p_neg_given_notD = 0.90
    p_pos_given_notD = 1.0 - p_neg_given_notD

    p_pos = p_pos_given_D * pD + p_pos_given_notD * (1.0 - pD)
    return (p_pos_given_D * pD) / p_pos


def q05() -> float:
    # E[X] = sum x*p(x)
    return 0 * 0.1 + 1 * 0.2 + 2 * 0.3 + 3 * 0.4


def q06() -> list[float]:
    # Binomial(n=4,p=0.3): P(X=k) = C(n,k)p^k(1-p)^(n-k)
    n, p = 4, 0.3
    return [math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in
range(n + 1)]


def q07() -> float:
    # Var of fair die: E[X^2] - (E[X])^2
    mu = 3.5
    return sum(((x - mu) ** 2) for x in [1, 2, 3, 4, 5, 6]) / 6


def q08() -> float:
    # Corr = Cov / (sdX * sdY)
    return 2 / (math.sqrt(9) * math.sqrt(16))


def q09() -> float:
    # P(8 < X < 12) for N(10,2)
    return float(norm.cdf(12, loc=10, scale=2) - norm.cdf(8, loc=10, scale=2))


def q10() -> float:
    # Margin of error = z * sigma / sqrt(n)
    return 1.96 * 4 / math.sqrt(25)


def q11() -> float:
    # two-sided p-value = 2*(1 - Phi(|z|))
    return float(2 * (1 - norm.cdf(2.1)))


def q12() -> float:
    # Monte Carlo estimate P(Z > 1.5)
    rng = np.random.default_rng(123)
    z = rng.standard_normal(200_000)
    return float((z > 1.5).mean())
