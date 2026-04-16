# Binary OU Cumulant Targets

## Setup

Binary OU process with hidden state $s_t \in \{+1, -1\}$ switching at rate $\kappa$. The observer sees only $X_t$ and must infer $s_t$ from data. We want to compute cumulants of the predictive distribution $p(X_{t+1} \mid X_{0:t})$ to arbitrary order.

## Predictive Distribution as a Gaussian Mixture

Given the trajectory $X_{0:t}$, the next-step state $s_{t+1}$ is either $+1$ or $-1$. Under each state, the OU transition gives a Gaussian with the same conditional variance but different means (pulling toward $+\mu$ or $-\mu$). Marginalizing over $s_{t+1}$ yields a two-component mixture:

$$p(X_{t+1} \mid X_{0:t}) = p_+\;\mathcal{N}(X_t c + \mu r,\;\sigma^2) \;+\; p_-\;\mathcal{N}(X_t c - \mu r,\;\sigma^2)$$

where $c = e^{-\theta\Delta t}$, $r = 1-c$, $\sigma^2 = (D/\theta)(1 - e^{-2\theta\Delta t})$, $p_+ = P(s_{t+1}=+1 \mid X_{0:t})$ from the HMM filter (see below), and $p_- = 1 - p_+$.

The two component means differ only in the sign of $\mu r$, and both share $\sigma^2$. This means sampling from the mixture is equivalent to:

$$X_{t+1} = \underbrace{X_t\, c}_{\text{deterministic decay}} + \underbrace{\delta\, S}_{\text{state-dependent pull}} + \underbrace{\varepsilon}_{\text{diffusion noise}}, \qquad S \perp \varepsilon$$

with $\delta = \mu r$, $S \in \{+1,-1\}$ with $P(S=+1) = p_+$, and $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. The three components are: a deterministic part (OU decay of the current position), a discrete random part (which regime the process is pulled toward), and a continuous Gaussian part (diffusion noise).

## General Cumulant Formula

By cumulant additivity:

$$\kappa_n(X_{t+1} \mid X_{0:t}) = \delta^n\,\kappa_n(S) + [n=1]\,X_t c + [n=2]\,\sigma^2$$

The raw moments of $S$ are trivial: $E[S^n] = 1$ (even $n$) or $2p_+ - 1$ (odd $n$). Cumulants of $\delta S$ then follow from the moment-cumulant recurrence:

$$\kappa_n = \mu'_n - \sum_{m=1}^{n-1}\binom{n-1}{m-1}\kappa_m\,\mu'_{n-m}$$

applied to $\mu'_k = \delta^k E[S^k]$. This is computable to arbitrary order with no case-by-case derivation.

**Explicit low-order results** (with $e = 2p_+ - 1$):

| $n$ | $\kappa_n$ |
|-----|------------|
| 1 | $X_t c + \delta e$ |
| 2 | $\delta^2(1 - e^2) + \sigma^2$ |
| 3 | $-2\delta^3 e(1 - e^2)$ |
| 4 | $-2\delta^4(1 - e^2)(1 - 3e^2)$ |

**Consistency check**: Standard OU (no switching) has $p_+ \in \{0,1\}$, so $e = \pm 1$, $1-e^2 = 0$, and all mixture cumulants vanish. We recover $\kappa_1 = X_t c \pm \mu r$ and $\kappa_2 = \sigma^2$.

## HMM Filter for $p_+$

### Why the filter is needed

The mixture weight $p_+ = P(s_{t+1} = +1 \mid X_{0:t})$ cannot be computed from $X_t$ alone. The hidden state must be inferred from the *full* observation history: a trajectory that has been hovering near $+\mu$ for many steps provides strong evidence that $s_t = +1$, even if the most recent $X_t$ happens to be near zero. This sequential inference is exactly what the HMM forward algorithm computes.

### The forward algorithm

We maintain the filtered belief $\pi_t = P(s_t = +1 \mid X_{0:t})$ and update it at each time step in two stages.

**Initialization**: $\pi_0 = 0.5$ (no prior preference for either state).

**Predict** — propagate through the state transition. Because the hidden chain can switch with probability $p_\text{sw} = 1 - e^{-\kappa\Delta t}$, the prior for the next state before seeing data is:

$$\pi^-_t = (1-p_\text{sw})\,\pi_{t-1} + p_\text{sw}\,(1-\pi_{t-1})$$

This shrinks the belief toward $0.5$: even if we were confident about the state at $t-1$, the possibility of a switch adds uncertainty.

**Update** — incorporate the new observation $X_t$ via Bayes' rule:

$$\pi_t = \frac{\ell_+\;\pi^-_t}{\ell_+\;\pi^-_t + \ell_-\;(1-\pi^-_t)}$$

where $\ell_\pm$ are the OU transition likelihoods under each state:

$$\ell_\pm = \mathcal{N}\!\bigl(X_t;\; X_{t-1}\,c \pm \mu\,r,\;\sigma^2\bigr)$$

An observation close to $X_{t-1}\,c + \mu\,r$ (the mean under $s=+1$) increases $\pi_t$; an observation close to $X_{t-1}\,c - \mu\,r$ decreases it.

### From filtered belief to predicted mixture weight

After filtering at time $t$, we have $\pi_t$ — the belief about the *current* state. But the cumulant formulas need $p_+$ — the predicted probability for the *next* state $s_{t+1}$, which accounts for the possible switch between $t$ and $t+1$:

$$p_+ = (1 - p_\text{sw})\,\pi_t + p_\text{sw}\,(1 - \pi_t)$$

Equivalently, defining the switching attenuation $b = 2e^{-\kappa\Delta t} - 1$:

$$e = 2p_+ - 1 = b\,(2\pi_t - 1)$$

The factor $b \in [0, 1]$ dampens the filtered belief: high switching rate ($\kappa$ large) drives $b \to 0$ and $p_+ \to 0.5$, reflecting near-complete uncertainty about the next state regardless of current evidence.

### Alternative: oracle targets

If we condition targets on the *true* hidden state $s_t$ instead of the filtered belief, the HMM filter is not needed. Within each regime the process is Gaussian, so $\kappa_{n \geq 3} = 0$ — the higher cumulants carry no information. However, the transformer still learns the HMM filter implicitly: under MSE loss on $\kappa_1$, the optimal predictor is $E[\kappa_1 \mid X_{0:t}]$, which equals the HMM-filtered conditional mean. This is a viable simpler alternative if only the first two cumulants are of interest.

## Proposed Code

```python
from math import comb

def compute_mixture_cumulants(p_plus, delta, sigma2, order):
    """Cumulants of delta*S + eps (S in {+1,-1}, eps ~ N(0, sigma2))."""
    raw_moments = []
    for n in range(1, order + 1):
        e_sn = p_plus + (1 - p_plus) * ((-1.0) ** n)
        raw_moments.append(delta ** n * e_sn)

    cumulants = []
    for n in range(order):
        kappa = raw_moments[n]
        for m in range(n):
            kappa = kappa - comb(n, m) * cumulants[m] * raw_moments[n - 1 - m]
        cumulants.append(kappa)

    if order >= 2:
        cumulants[1] = cumulants[1] + sigma2
    return cumulants
```

Replaces the four hand-coded cumulant blocks with a single general function that supports arbitrary `order`.
