# Mathematical Theory: Wavelet Triplane Diffusion

## Table of Contents
1. [Overview](#overview)
2. [Wavelet Transform Theory](#wavelet-transform-theory)
3. [Triplane Representation](#triplane-representation)
4. [Diffusion Models](#diffusion-models)
5. [Complete Pipeline](#complete-pipeline)
6. [Training Objectives](#training-objectives)

---

## Overview

This architecture combines three powerful techniques:
- **Wavelet Decomposition**: Multi-resolution signal analysis
- **Triplane Representation**: Efficient 3D encoding
- **Diffusion Models**: Generative modeling for high-frequency details

The key insight: Generate **structure** (low-frequency) cheaply, then use diffusion to add **details** (high-frequency).

---

## Wavelet Transform Theory

### 1. Discrete Wavelet Transform (2D)

For an image $I \in \mathbb{R}^{H \times W}$, the 2D Haar wavelet transform decomposes it into 4 subbands:

$$
\begin{aligned}
LL_{i,j} &= \frac{1}{2}(I_{2i,2j} + I_{2i+1,2j} + I_{2i,2j+1} + I_{2i+1,2j+1}) \\
LH_{i,j} &= \frac{1}{2}(I_{2i,2j} + I_{2i+1,2j} - I_{2i,2j+1} - I_{2i+1,2j+1}) \\
HL_{i,j} &= \frac{1}{2}(I_{2i,2j} - I_{2i+1,2j} + I_{2i,2j+1} - I_{2i+1,2j+1}) \\
HH_{i,j} &= \frac{1}{2}(I_{2i,2j} - I_{2i+1,2j} - I_{2i,2j+1} + I_{2i+1,2j+1})
\end{aligned}
$$

Where:
- $LL$: Low-Low (approximation) - contains the overall structure
- $LH$: Low-High (horizontal details) - vertical edges
- $HL$: High-Low (vertical details) - horizontal edges
- $HH$: High-High (diagonal details) - corners and textures

Each subband has resolution $\frac{H}{2} \times \frac{W}{2}$.

### 2. Inverse Wavelet Transform (IDWT)

The reconstruction formula:

$$
I_{2i,2j} = \frac{1}{2}(LL + LH + HL + HH)
$$
$$
I_{2i+1,2j} = \frac{1}{2}(LL + LH - HL - HH)
$$
$$
I_{2i,2j+1} = \frac{1}{2}(LL - LH + HL - HH)
$$
$$
I_{2i+1,2j+1} = \frac{1}{2}(LL - LH - HL + HH)
$$

This is **perfectly invertible**: $IDWT(DWT(I)) = I$ (up to numerical precision).

### 3. Matrix Form

Using convolution filters:

$$
\begin{bmatrix}
LL \\
LH \\
HL \\
HH
\end{bmatrix} = 
\begin{bmatrix}
h_0 \otimes h_0 \\
h_0 \otimes h_1 \\
h_1 \otimes h_0 \\
h_1 \otimes h_1
\end{bmatrix} * I
$$

Where:
- $h_0 = \frac{1}{\sqrt{2}}[1, 1]$ (low-pass filter)
- $h_1 = \frac{1}{\sqrt{2}}[1, -1]$ (high-pass filter)
- $\otimes$ denotes outer product
- $*$ denotes 2D convolution with stride 2

### 4. Implementation in PyTorch

```python
# Forward DWT
filters = create_haar_filters()  # Shape: (4, 1, 2, 2)
output = F.conv2d(x, filters, stride=2)  # Downsample by 2

# Inverse IDWT
x_recon = F.conv_transpose2d(bands, filters, stride=2)  # Upsample by 2
```

---

## Triplane Representation

### 1. Definition

A triplane $\mathcal{T} = \{T_{XY}, T_{XZ}, T_{YZ}\}$ consists of 3 feature maps:

$$
T_{XY} \in \mathbb{R}^{C \times H \times W}, \quad T_{XZ} \in \mathbb{R}^{C \times H \times W}, \quad T_{YZ} \in \mathbb{R}^{C \times H \times W}
$$

Each plane stores features for 2D slices through 3D space.

### 2. Feature Query

For a 3D point $\mathbf{p} = (x, y, z) \in [-1, 1]^3$:

$$
\mathbf{f}(\mathbf{p}) = \text{Concat}\Big(
    T_{XY}(x, y), \quad
    T_{XZ}(x, z), \quad
    T_{YZ}(y, z)
\Big) \in \mathbb{R}^{3C}
$$

Where $T(\cdot, \cdot)$ is bilinear interpolation:

$$
T(u, v) = \sum_{i,j} w_{ij} \cdot T[i, j]
$$

with interpolation weights $w_{ij}$ based on $(u, v)$.

### 3. Decoding

A small MLP decodes features to RGB + density:

$$
\begin{aligned}
(\mathbf{c}, \sigma) &= \text{MLP}(\mathbf{f}(\mathbf{p})) \\
\mathbf{c} &\in \mathbb{R}^3 \quad \text{(RGB color)} \\
\sigma &\in \mathbb{R}^+ \quad \text{(density)}
\end{aligned}
$$

### 4. Volume Rendering

Using the emission-absorption model:

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t)) \, dt
$$

Where:
- $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ is the ray
- $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)$ is transmittance
- Discretized via quadrature: $C \approx \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i$

---

## Diffusion Models

### 1. Forward Process (Adding Noise)

Given clean data $\mathbf{x}_0$, progressively add Gaussian noise:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

Using the reparameterization trick:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$

Where $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$.

### 2. Reverse Process (Denoising)

Learn to predict the noise:

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) \approx \boldsymbol{\epsilon}
$$

Where $\mathbf{c}$ is conditioning information (e.g., LL band).

### 3. Training Objective

Simplified objective (Ho et al., 2020):

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})\|^2 \right]
$$

### 4. Sampling (DDPM)

Iteratively denoise from random noise:

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) \right) + \sigma_t \mathbf{z}
$$

Where $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ and $\sigma_t = \sqrt{\beta_t}$.

### 5. Faster Sampling (DDIM)

Deterministic sampling (Song et al., 2020):

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{predicted } \mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_\theta
$$

Allows 10-50 steps instead of 1000.

---

## Complete Pipeline

### Stage 1: Coarse Triplane Generation

$$
T_{LL} = E_\phi(I)
$$

Where:
- $I \in \mathbb{R}^{3 \times 256 \times 256}$ is the input image
- $E_\phi$ is a CNN/ViT encoder
- $T_{LL} \in \mathbb{R}^{3 \times C \times 64 \times 64}$ is the low-res triplane (LL band)

### Stage 2: Wavelet Detail Diffusion

For each plane $p \in \{XY, XZ, YZ\}$:

$$
\mathbf{d}_p = \{LH_p, HL_p, HH_p\} = \text{Denoise}_\theta(T_{LL,p})
$$

Where:
- $T_{LL,p} \in \mathbb{R}^{C \times 64 \times 64}$ is the conditioning
- $\mathbf{d}_p \in \mathbb{R}^{3C \times 64 \times 64}$ are the predicted high-freq details
- $\text{Denoise}_\theta$ is the conditional U-Net diffusion model

Training loss for this stage:

$$
\mathcal{L}_{\text{diff}} = \sum_{p} \mathbb{E}_{t, \mathbf{d}_p, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta([\mathbf{d}_p]_t, t, T_{LL,p})\|^2 \right]
$$

### Stage 3: Inverse Wavelet Transform

For each plane $p$:

$$
T_{HR,p} = \text{IDWT}(T_{LL,p}, LH_p, HL_p, HH_p)
$$

Resulting in:
$$
T_{HR} \in \mathbb{R}^{3 \times C \times 128 \times 128}
$$

This doubles the resolution: $64 \times 64 \to 128 \times 128$.

### Stage 4: Volume Rendering

$$
I_{\text{render}} = R(T_{HR})
$$

Where $R$ is the volumetric rendering function using the triplane decoder.

---

## Training Objectives

### 1. Full Loss Function

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{diff}} \mathcal{L}_{\text{diff}} + \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \lambda_{\text{render}} \mathcal{L}_{\text{render}}
$$

#### a) Diffusion Loss

$$
\mathcal{L}_{\text{diff}} = \sum_{p=1}^{3} \mathbb{E}_{t, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{d}_{p,t}, t, T_{LL,p})\|^2 \right]
$$

#### b) Reconstruction Loss

If ground truth high-res triplane $T_{GT}$ is available:

$$
\mathcal{L}_{\text{recon}} = \|T_{HR} - T_{GT}\|_1
$$

Using L1 loss for sharper details.

#### c) Rendering Loss

Given target views $I_{target}$:

$$
\mathcal{L}_{\text{render}} = \|I_{\text{render}} - I_{target}\|_2^2 + \lambda_{\text{LPIPS}} \mathcal{L}_{\text{LPIPS}}(I_{\text{render}}, I_{target})
$$

Where $\mathcal{L}_{\text{LPIPS}}$ is perceptual loss.

### 2. Two-Stage Training

**Stage 1: Pre-train Coarse Generator**

$$
\min_\phi \mathcal{L}_{\text{coarse}} = \|T_{LL} - T_{GT,LL}\|_2^2 + \alpha \mathcal{L}_{\text{render}}(E_\phi(I))
$$

**Stage 2: Train Diffusion Model**

Fix $E_\phi$, train diffusion:

$$
\min_\theta \mathcal{L}_{\text{diff}}
$$

**Stage 3: Fine-tune End-to-End**

Jointly optimize all components:

$$
\min_{\phi, \theta} \mathcal{L}_{\text{total}}
$$

### 3. Practical Considerations

**Noise Schedule**: Use cosine schedule for better high-res generation:

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2
$$

**Conditioning**: Inject $T_{LL}$ into U-Net via:
- Cross-attention: $\text{Attn}(Q=h, K=T_{LL}, V=T_{LL})$
- Concatenation: $h' = \text{Conv}([h, T_{LL}])$

**Classifier-Free Guidance**: During sampling, use:

$$
\tilde{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + w \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, T_{LL}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset))
$$

Where $w > 1$ is guidance scale.

---

## Theoretical Advantages

### 1. Computational Efficiency

**Memory Scaling**:
- Direct generation at 128×128: $O(128^2) = 16,384$ pixels
- Wavelet approach: $O(64^2 + 64^2) = 8,192$ (50% savings)

**Computation Scaling**:
- U-Net at 64×64 is ~4× faster than at 128×128
- Diffusion steps only on details (3/4 of the data)

### 2. Representation Quality

**Frequency Decomposition**:
- LL band captures global structure (easy to predict)
- High-freq bands capture fine details (hard, but smaller)
- Diffusion excels at details/textures

**Progressive Generation**:
- Coarse-to-fine naturally matches human perception
- Can generate structure quickly, refine later

### 3. Multi-Resolution Training

Can train at different resolutions:
- Level 1: $32 \to 64$
- Level 2: $64 \to 128$
- Level 3: $128 \to 256$

Cascaded diffusion for extreme resolutions.

---

## Connections to Related Work

1. **EG3D (Chan et al., 2022)**: Triplane representation
2. **Latent Diffusion (Rombach et al., 2022)**: Diffusion in compressed space
3. **Cascaded Diffusion (Ho et al., 2022)**: Multi-resolution generation
4. **SinGAN (Shaham et al., 2019)**: Pyramid-based generation

**Our Contribution**: Combines wavelets + triplanes + diffusion in a unified framework optimized for 3D generation.

---

## Future Directions

1. **Multi-Scale Wavelets**: Extend to 3+ levels (512×512+)
2. **3D Wavelets**: Apply DWT to the full 3D volume, not just planes
3. **Learned Wavelets**: Replace Haar with learned filters
4. **Adaptive Decomposition**: Predict which regions need high-freq details
5. **Video Extension**: Temporal wavelets for 4D generation

---

## References

1. Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models.
2. Song, J., et al. (2020). Denoising Diffusion Implicit Models.
3. Chan, E., et al. (2022). Efficient Geometry-aware 3D Generative Adversarial Networks (EG3D).
4. Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.
5. Mallat, S. (1989). A Theory for Multiresolution Signal Decomposition: The Wavelet Representation.

---

**End of Mathematical Theory Document**
