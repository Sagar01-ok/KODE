# Diffusion Mathematics & Sampling Algorithms

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Variance Schedules, DDPM Ancestral Sampling, DDIM Accelerated Sampling  

---

## 1. Variance Schedules

KODE implements both Linear and Cosine variance schedules.

### 1.1 Linear Variance Schedule (Ho et al., 2020)
$$\beta_t = \beta_{\text{start}} + \frac{t - 1}{T - 1} (\beta_{\text{end}} - \beta_{\text{start}})$$
Default parameters: $T = 1000, \beta_{\text{start}} = 10^{-4}, \beta_{\text{end}} = 0.02$.

### 1.2 Cosine Variance Schedule (Nichol & Dhariwal, 2021)
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2, \quad s = 0.008$$
$$\beta_t = \text{clip}\left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.0, 0.999\right)$$
The cosine schedule prevents abrupt information loss at early timesteps, providing superior contrast and detail for low-resolution generation.

---

## 2. Reverse Sampling Algorithms

### 2.1 DDPM Ancestral Sampling
* Generates samples by stepping backwards from $t=T$ down to $t=1$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \tilde{\epsilon}_\theta \right) + \sigma_t z$$
where $z \sim \mathcal{N}(0, \mathbf{I})$ for $t > 1$, and $z = 0$ for $t = 1$.
* Execution time on CPU: ~12 seconds for $T=1000$.

### 2.2 DDIM Accelerated Deterministic Sampling (Song et al., 2020)
* Selects a sub-sequence of $S$ evaluation timesteps $\mathcal{T} = \{\tau_1, \tau_2, \dots, \tau_S\}$ (e.g. $S = 25$ steps):
$$\hat{x}_0 = \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \tilde{\epsilon}_\theta}{\sqrt{\bar{\alpha}_{\tau_i}}}$$
$$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}}} \tilde{\epsilon}_\theta$$
* Execution time on CPU: ~0.3 seconds for $S=25$.
* Trajectories are strictly deterministic given a fixed random seed.
