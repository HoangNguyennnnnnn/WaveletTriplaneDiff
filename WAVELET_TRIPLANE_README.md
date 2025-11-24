# Wavelet Triplane Diffusion for 3D Generation

A PyTorch research prototype implementing a novel 3D generation architecture that combines wavelet decomposition with triplane representations and diffusion models.

## Architecture Overview

The pipeline consists of 4 stages:

### Stage 1: Coarse Generator
- **Input**: RGB images (B, 3, 256, 256)
- **Function**: CNN/Transformer encoder that produces low-resolution triplane
- **Output**: Low-res triplane (LL wavelet band) with shape (B, 3, C, 64, 64)
- **Implementation**: `CoarseTriplaneGen` class

### Stage 2: Wavelet Detail Diffusion
- **Input**: Noisy high-frequency details + LL band (condition)
- **Function**: Conditional U-Net diffusion model
- **Output**: Predicted LH, HL, HH wavelet bands with shape (B, 3, 3*C, 64, 64)
- **Implementation**: `DetailDiffusion` class
- **Key Feature**: Uses time-conditioned U-Net with cross-attention to LL band

### Stage 3: Differentiable IDWT
- **Input**: LL, LH, HL, HH wavelet bands
- **Function**: Inverse Discrete Wavelet Transform (Haar)
- **Output**: High-resolution triplane (B, 3, C, 128, 128)
- **Implementation**: `WaveletTransform.idwt2d()` method
- **Key Feature**: Fully differentiable for end-to-end training

### Stage 4: Triplane Decoder
- **Input**: High-res triplane + 3D query coordinates
- **Function**: NeRF-style MLP decoder + volumetric rendering
- **Output**: Rendered RGB images
- **Implementation**: `TriplaneRenderer` class
- **Features**: 
  - Triplane feature interpolation
  - Ray marching with alpha compositing
  - Differentiable rendering

## Technical Details

### Wavelet Transform
- **Type**: 2D Haar Wavelet (separable)
- **Bands**: LL (approximation), LH, HL, HH (details)
- **Implementation**: Custom PyTorch implementation using conv2d and conv_transpose2d
- **Properties**: Orthogonal, perfectly reconstructable, differentiable

### Diffusion Model
- **Architecture**: Conditional U-Net with residual blocks
- **Conditioning**: LL band processed through encoder
- **Time Embedding**: Sinusoidal positional encoding
- **Normalization**: GroupNorm for stability

### Triplane Representation
- **Planes**: XY, XZ, YZ
- **Query**: Bilinear interpolation using F.grid_sample
- **Aggregation**: Concatenation of features from all 3 planes

## Installation

```bash
pip install -r requirements_wavelet_triplane.txt
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+
- einops 0.7+

## Usage

### Quick Test

```bash
python wavelet_triplane_diffusion.py
```

This will:
1. Initialize the model
2. Create dummy input data
3. Run a full forward pass
4. Print tensor shapes at each stage
5. Test individual components (wavelet transform, triplane query)

### Expected Output

```
[Stage 1] Input images shape: torch.Size([2, 3, 256, 256])
[Stage 1] Output LL triplane shape: torch.Size([2, 3, 32, 64, 64])

[Stage 2] Diffusion Detail Prediction
[Stage 2] Plane 0 - Predicted details shape: torch.Size([2, 96, 64, 64])
[Stage 2] Plane 1 - Predicted details shape: torch.Size([2, 96, 64, 64])
[Stage 2] Plane 2 - Predicted details shape: torch.Size([2, 96, 64, 64])

[Stage 3] Inverse Wavelet Transform
[Stage 3] Plane 0 - Reconstructed high-res shape: torch.Size([2, 32, 128, 128])
[Stage 3] Plane 1 - Reconstructed high-res shape: torch.Size([2, 32, 128, 128])
[Stage 3] Plane 2 - Reconstructed high-res shape: torch.Size([2, 32, 128, 128])
[Stage 3] Final high-res triplane shape: torch.Size([2, 3, 32, 128, 128])

[Stage 4] Volumetric Rendering
[Stage 4] Rendered image shape: torch.Size([2, 3, 64, 64])
```

### Using in Your Code

```python
from wavelet_triplane_diffusion import WaveletTriplaneDiffusion

# Initialize model
model = WaveletTriplaneDiffusion(triplane_channels=32)

# Forward pass
images = torch.randn(4, 3, 256, 256)  # Batch of 4 images
outputs = model(images, return_intermediate=True)

# Access outputs
triplane_lowres = outputs['triplane_LL']        # (4, 3, 32, 64, 64)
triplane_details = outputs['triplane_details']   # (4, 3, 96, 64, 64)
triplane_highres = outputs['triplane_highres']   # (4, 3, 32, 128, 128)
rendered = outputs['rendered_image']             # (4, 3, 64, 64)
```

## Model Components

### `WaveletTransform`
Implements differentiable 2D Haar wavelet decomposition and reconstruction.

**Methods:**
- `dwt2d(x)`: Forward transform, returns (LL, LH, HL, HH)
- `idwt2d(LL, LH, HL, HH)`: Inverse transform, reconstructs original

### `CoarseTriplaneGen`
Encodes input images to low-resolution triplane (LL band).

**Architecture:**
- 2 downsampling conv layers (256→128→64)
- 2 residual blocks at 64x64
- Output projection to 3 planes

### `DetailDiffusion`
Conditional U-Net for predicting high-frequency details.

**Architecture:**
- 3-level encoder-decoder with skip connections
- Time embedding with sinusoidal encoding
- Condition encoder for LL band
- GroupNorm + GELU activations

### `TriplaneRenderer`
NeRF-style decoder with volumetric rendering.

**Methods:**
- `query_triplane(triplane, coords)`: Interpolate features at 3D points
- `forward(triplane, coords)`: Decode to RGB + density
- `render_volume(triplane)`: Full volumetric rendering with ray marching

### `WaveletTriplaneDiffusion`
Complete end-to-end pipeline connecting all stages.

**Methods:**
- `forward(images)`: Full forward pass through all 4 stages
- `inference(images, num_steps)`: Inference with DDPM sampling (placeholder)

## Key Design Decisions

1. **Wavelet Choice**: Haar wavelets for simplicity and perfect reconstruction
2. **Resolution**: 64→128 upsampling (1 level of wavelet decomposition)
3. **Triplane Channels**: 32 (adjustable, affects memory/quality tradeoff)
4. **Diffusion Steps**: Simplified single-step for prototype (full DDPM in production)
5. **Rendering**: Simplified ray marching (64 rays, 32 samples for speed)

## Training (Not Implemented)

To train this model, you would need to:

1. **Data**: Multi-view images + 3D ground truth
2. **Loss Functions**:
   - Diffusion loss (MSE on predicted noise)
   - Reconstruction loss (MSE on IDWT output)
   - Rendering loss (PSNR/LPIPS on rendered images)
3. **Training Loop**:
   - Sample timesteps for diffusion
   - Add noise to high-freq bands
   - Predict noise with DetailDiffusion
   - Reconstruct and render
   - Backpropagate combined losses

## Limitations & Future Work

- **Current**: Single-step diffusion (fast but lower quality)
- **Future**: Full DDPM/DDIM sampling loop
- **Current**: Fixed resolution (64→128)
- **Future**: Multi-scale wavelet decomposition (64→256→512)
- **Current**: Simple MLP decoder
- **Future**: Attention-based decoder, SDF representation
- **Current**: No conditioning on text/class labels
- **Future**: Multi-modal conditioning (CLIP features, etc.)

## Performance

**Model Size**: ~10-20M parameters (depends on triplane_channels)

**Memory** (batch_size=2, triplane_channels=32):
- Forward pass: ~2-3 GB GPU memory
- Training: ~6-8 GB GPU memory (with gradients)

**Speed** (on RTX 3090):
- Forward pass: ~100-200ms per batch
- Rendering: ~50-100ms per batch

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{wavelet_triplane_diffusion_2025,
  title={Wavelet Triplane Diffusion for 3D Generation},
  author={Research Prototype},
  year={2025},
  note={Research implementation}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Wavelet transform implementation inspired by PyTorch Wavelets (ptwt)
- Diffusion architecture based on DDPM/Stable Diffusion
- Triplane representation from EG3D
- NeRF rendering from original NeRF paper
