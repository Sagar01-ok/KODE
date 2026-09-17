# Mathematical Foundations of KODE

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Formal Mathematical Formulations & Derivations  

---

## 1. Tensor Operations and Calculus

### 1.1 Matrix Multiplication and Adjoint Gradients
Given input activation $X \in \mathbb{R}^{M \times K}$ and weight matrix $W \in \mathbb{R}^{K \times N}$, the forward linear transformation with bias $b \in \mathbb{R}^{N}$ is:
$$Y = X W + \mathbf{1} b^T \in \mathbb{R}^{M \times N}$$

During reverse-mode automatic differentiation, given upstream gradient $\nabla_Y \mathcal{L} = \frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{M \times N}$, the adjoint derivatives with respect to inputs and parameters are:
$$\nabla_X \mathcal{L} = (\nabla_Y \mathcal{L}) W^T \in \mathbb{R}^{M \times K}$$
$$\nabla_W \mathcal{L} = X^T (\nabla_Y \mathcal{L}) \in \mathbb{R}^{K \times N}$$
$$\nabla_b \mathcal{L} = \sum_{i=1}^M (\nabla_Y \mathcal{L})_{i, :} \in \mathbb{R}^{N}$$

### 1.2 2D Convolution Forward and Backward
For input tensor $X \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$ and filter kernel $K \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$:
The forward spatial convolution (cross-correlation) at output position $(b, c_{\text{out}}, i, j)$ with stride $s$ and padding $p$ is:
$$Y_{b, c_{\text{out}}, i, j} = \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} X_{b, c_{\text{in}}, i \cdot s + u - p, j \cdot s + v - p} K_{c_{\text{out}}, c_{\text{in}}, u, v} + b_{c_{\text{out}}}$$

Using the `im2col` transformation, the 4D convolution maps directly to a 2D matrix multiplication:
$$Y_{\text{col}} = K_{\text{flat}} \cdot X_{\text{col}}$$
This allows the backward passes $\nabla_X \mathcal{L}$ and $\nabla_K \mathcal{L}$ to be computed via transposed GEMM:
$$\nabla_{K_{\text{flat}}} \mathcal{L} = (\nabla_{Y_{\text{col}}} \mathcal{L}) \cdot X_{\text{col}}^T$$
$$\nabla_{X_{\text{col}}} \mathcal{L} = K_{\text{flat}}^T \cdot (\nabla_{Y_{\text{col}}} \mathcal{L})$$
followed by the inverse `col2im` accumulation.

### 1.3 Group Normalization Calculus
For a feature vector $x$ within a group $G$ of channels ($|G| = \frac{C}{\text{num\_groups}} \times H \times W$ elements):
$$\mu = \frac{1}{|G|} \sum_{i \in G} x_i, \quad \sigma^2 = \frac{1}{|G|} \sum_{i \in G} (x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
Upstream gradient propagation yields:
$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma$$
$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{|G| \sqrt{\sigma^2 + \epsilon}} \left( |G| \frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \sum_{j \in G} \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \hat{x}_i \sum_{j \in G} \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j \right)$$

---

## 2. Denoising Diffusion Probabilistic Mathematics

### 2.1 The Forward Markov Chain
The forward corruption process adds Gaussian noise according to variance schedule $\beta_1, \beta_2, \dots, \beta_T$:
$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}\right)$$

Defining $\alpha_t = 1 - \beta_t$ and cumulative product $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:
$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1}$$
Unrolling the recurrence recursively:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
This closed form enables training at arbitrary timestep $t$ in $O(1)$ time without simulating steps $1, \dots, t-1$.

### 2.2 Reverse Posterior and Objective Derivation
Using Bayes' rule, the posterior conditioned on $x_0$ is:
$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)} = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})$$
where:
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

Since $x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}$, substituting into $\tilde{\mu}_t$:
$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$$

Training our neural network $\epsilon_\theta(x_t, t, c)$ to predict noise $\epsilon$ optimizes the variational bound on negative log-likelihood:
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t \sim [1, T], x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]$$

### 2.3 Sampling Formulations

#### 2.3.1 DDPM Stochastic Ancestral Sampling
For $t = T, T-1, \dots, 1$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I}) \text{ if } t > 1 \text{ else } 0$$
where $\sigma_t = \sqrt{\tilde{\beta}_t}$.

#### 2.3.2 DDIM Deterministic Sampling
Song et al. (2020) generalized the reverse process to non-Markovian inference while matching the identical forward marginals $q(x_t|x_0)$. With stochasticity parameter $\eta = 0$, the reverse update becomes strictly deterministic:
$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t, c)}{\sqrt{\bar{\alpha}_t}}$$
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t, c)$$
This enables accelerated sampling across a sub-sequence of steps $\{\tau_1, \dots, \tau_S\}$ (e.g., $S=25$).

### 2.4 Classifier-Free Guidance (CFG)
To steer generation strongly towards prompt conditioning $c$:
$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))$$
where:
* $\emptyset$ is the unconditional embedding (empty string / token ID 4).
* $s \ge 1.0$ is the guidance scale (typically $s \in [3.0, 7.5]$).

---

## 3. AdamW Optimizer Formulation
At optimization step $k$ with parameters $\theta_k$, learning rate $\eta$, weight decay $\lambda$, and loss gradient $g_k = \nabla_\theta \mathcal{L}$:

$$m_k = \beta_1 m_{k-1} + (1 - \beta_1) g_k \quad \text{(1st moment)}$$
$$v_k = \beta_2 v_{k-1} + (1 - \beta_2) g_k^2 \quad \text{(2nd uncentered moment)}$$
$$\hat{m}_k = \frac{m_k}{1 - \beta_1^k}, \quad \hat{v}_k = \frac{v_k}{1 - \beta_2^k} \quad \text{(Bias corrections)}$$
$$\theta_{k+1} = \theta_k - \eta \left( \frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon} + \lambda \theta_k \right) \quad \text{(Decoupled weight decay update)}$$
Typical hyperparameter settings: $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}, \lambda = 0.01$.
