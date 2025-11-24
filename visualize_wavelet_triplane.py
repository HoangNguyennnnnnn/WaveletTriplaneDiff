"""
Visualization Script for Wavelet Triplane Diffusion

This script helps visualize:
1. Wavelet decomposition (LL, LH, HL, HH bands)
2. Triplane structure
3. Rendered outputs

Requires: matplotlib
Install with: pip install matplotlib
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wavelet_triplane_diffusion import (
    WaveletTriplaneDiffusion, 
    WaveletTransform,
    CoarseTriplaneGen,
    TriplaneRenderer
)


def visualize_wavelet_decomposition(image: torch.Tensor, wavelet: WaveletTransform):
    """
    Visualize 2D wavelet decomposition of a single-channel image.
    
    Args:
        image: Input tensor of shape (1, C, H, W)
        wavelet: WaveletTransform instance
    """
    # Decompose
    LL, LH, HL, HH = wavelet.dwt2d(image)
    
    # Convert to numpy
    LL = LL[0, 0].cpu().numpy()
    LH = LH[0, 0].cpu().numpy()
    HL = HL[0, 0].cpu().numpy()
    HH = HH[0, 0].cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(LL, cmap='viridis')
    axes[0, 0].set_title('LL (Approximation - Low-Low)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(LH, cmap='viridis')
    axes[0, 1].set_title('LH (Horizontal Details - Low-High)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(HL, cmap='viridis')
    axes[1, 0].set_title('HL (Vertical Details - High-Low)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(HH, cmap='viridis')
    axes[1, 1].set_title('HH (Diagonal Details - High-High)')
    axes[1, 1].axis('off')
    
    plt.suptitle('2D Haar Wavelet Decomposition', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('wavelet_decomposition.png', dpi=150, bbox_inches='tight')
    print("Saved: wavelet_decomposition.png")
    plt.close()


def visualize_triplane_structure(triplane: torch.Tensor):
    """
    Visualize the 3-plane structure.
    
    Args:
        triplane: Triplane tensor of shape (1, 3, C, H, W)
    """
    # Extract planes (visualize first channel)
    xy_plane = triplane[0, 0, 0].cpu().numpy()
    xz_plane = triplane[0, 1, 0].cpu().numpy()
    yz_plane = triplane[0, 2, 0].cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(xy_plane, cmap='coolwarm')
    ax1.set_title('XY Plane (Top View)', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(xz_plane, cmap='coolwarm')
    ax2.set_title('XZ Plane (Front View)', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(yz_plane, cmap='coolwarm')
    ax3.set_title('YZ Plane (Side View)', fontweight='bold')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    
    plt.suptitle('Triplane Representation (3 Orthogonal Planes)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('triplane_structure.png', dpi=150, bbox_inches='tight')
    print("Saved: triplane_structure.png")
    plt.close()


def visualize_pipeline_outputs(outputs: dict):
    """
    Visualize outputs from each stage of the pipeline.
    
    Args:
        outputs: Dictionary from model forward pass
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Input image
    input_img = outputs['input_images'][0].permute(1, 2, 0).cpu().numpy()
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_img)
    ax1.set_title('Input Image', fontweight='bold')
    ax1.axis('off')
    
    # Low-res triplane (LL band) - visualize first channel of each plane
    triplane_LL = outputs['triplane_LL'][0]  # (3, C, 64, 64)
    
    for i in range(3):
        ax = fig.add_subplot(gs[0, i+1])
        plane = triplane_LL[i, 0].cpu().numpy()
        im = ax.imshow(plane, cmap='viridis')
        ax.set_title(f'LL Plane {i} (64x64)', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # High-freq details - visualize LH band from each plane
    triplane_details = outputs['triplane_details'][0]  # (3, C*3, 64, 64)
    
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        # Extract LH band (first C channels)
        C = triplane_details.shape[1] // 3
        lh_band = triplane_details[i, :C].mean(dim=0).cpu().numpy()
        im = ax.imshow(lh_band, cmap='RdBu_r')
        ax.set_title(f'LH Details Plane {i}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # High-res triplane - visualize first channel
    triplane_highres = outputs['triplane_highres'][0]  # (3, C, 128, 128)
    
    ax = fig.add_subplot(gs[1, 3])
    highres_vis = triplane_highres[0, 0].cpu().numpy()
    im = ax.imshow(highres_vis, cmap='viridis')
    ax.set_title('High-Res Plane 0 (128x128)', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Rendered output
    rendered = outputs['rendered_image'][0].permute(1, 2, 0).cpu().numpy()
    rendered = np.clip(rendered, 0, 1)
    
    ax = fig.add_subplot(gs[2, :2])
    ax.imshow(rendered)
    ax.set_title('Rendered Output (Volume Rendering)', fontweight='bold')
    ax.axis('off')
    
    # Add text annotations
    ax_text = fig.add_subplot(gs[2, 2:])
    ax_text.axis('off')
    info_text = f"""
    Pipeline Summary:
    
    Stage 1: Coarse Generator
      Input: {tuple(outputs['input_images'].shape)}
      Output LL: {tuple(outputs['triplane_LL'].shape)}
    
    Stage 2: Detail Diffusion
      Predicted Details: {tuple(outputs['triplane_details'].shape)}
      (LH, HL, HH bands)
    
    Stage 3: IDWT Reconstruction
      High-Res Triplane: {tuple(outputs['triplane_highres'].shape)}
      Resolution: 64 → 128 (2x upsampling)
    
    Stage 4: Volume Rendering
      Output: {tuple(outputs['rendered_image'].shape)}
    """
    ax_text.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center')
    
    plt.suptitle('Wavelet Triplane Diffusion - Pipeline Visualization', 
                 fontsize=18, fontweight='bold')
    plt.savefig('pipeline_outputs.png', dpi=150, bbox_inches='tight')
    print("Saved: pipeline_outputs.png")
    plt.close()


def compare_resolutions(model: WaveletTriplaneDiffusion):
    """
    Compare low-res vs high-res triplane quality.
    """
    device = next(model.parameters()).device
    
    # Create test input
    test_image = torch.randn(1, 3, 256, 256, device=device)
    
    with torch.no_grad():
        outputs = model(test_image, return_intermediate=True)
    
    # Extract triplanes
    triplane_LL = outputs['triplane_LL'][0, 0, 0].cpu().numpy()  # 64x64
    triplane_HR = outputs['triplane_highres'][0, 0, 0].cpu().numpy()  # 128x128
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(triplane_LL, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Low-Res (LL Band Only)\n64 × 64', fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(triplane_HR, cmap='viridis', interpolation='nearest')
    axes[1].set_title('High-Res (After IDWT)\n128 × 128', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle('Resolution Comparison: Wavelet Upsampling', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('resolution_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: resolution_comparison.png")
    plt.close()


def visualize_frequency_separation():
    """
    Demonstrate how wavelets separate low and high frequencies.
    """
    # Create test signal with mixed frequencies
    x = np.linspace(0, 4*np.pi, 256)
    y = np.linspace(0, 4*np.pi, 256)
    X, Y = np.meshgrid(x, y)
    
    # Low frequency component
    low_freq = np.sin(X) * np.cos(Y)
    
    # High frequency component
    high_freq = 0.3 * np.sin(5*X) * np.cos(5*Y)
    
    # Combined signal
    signal = low_freq + high_freq
    
    # Apply wavelet transform
    signal_torch = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)
    wavelet = WaveletTransform()
    LL, LH, HL, HH = wavelet.dwt2d(signal_torch)
    
    # Convert back to numpy
    LL = LL[0, 0].numpy()
    LH = LH[0, 0].numpy()
    HL = HL[0, 0].numpy()
    HH = HH[0, 0].numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original components
    im1 = axes[0, 0].imshow(low_freq, cmap='seismic')
    axes[0, 0].set_title('Low Frequency Component', fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(high_freq, cmap='seismic')
    axes[0, 1].set_title('High Frequency Component', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(signal, cmap='seismic')
    axes[0, 2].set_title('Combined Signal', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Wavelet bands
    im4 = axes[1, 0].imshow(LL, cmap='seismic')
    axes[1, 0].set_title('LL (Captures Low Freq)', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(LH, cmap='seismic')
    axes[1, 1].set_title('LH (Captures Horizontal High Freq)', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Combine high-freq bands
    high_freq_combined = np.sqrt(LH**2 + HL**2 + HH**2)
    im6 = axes[1, 2].imshow(high_freq_combined, cmap='seismic')
    axes[1, 2].set_title('Combined High-Freq Bands', fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.suptitle('Wavelet Transform: Frequency Separation', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('frequency_separation.png', dpi=150, bbox_inches='tight')
    print("Saved: frequency_separation.png")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("Wavelet Triplane Diffusion - Visualization Suite")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")
    
    # 1. Frequency separation demo
    print("[1/5] Creating frequency separation visualization...")
    visualize_frequency_separation()
    
    # 2. Wavelet decomposition
    print("[2/5] Creating wavelet decomposition visualization...")
    wavelet = WaveletTransform()
    test_image = torch.randn(1, 1, 128, 128)
    visualize_wavelet_decomposition(test_image, wavelet)
    
    # 3. Initialize model
    print("[3/5] Initializing model...")
    model = WaveletTriplaneDiffusion(triplane_channels=32)
    model = model.to(device)
    model.eval()
    
    # 4. Generate outputs
    print("[4/5] Running forward pass...")
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        outputs = model(dummy_input, return_intermediate=True)
    
    # 5. Visualize pipeline
    print("[5/5] Creating pipeline visualization...")
    visualize_pipeline_outputs(outputs)
    visualize_triplane_structure(outputs['triplane_LL'])
    compare_resolutions(model)
    
    print("\n" + "=" * 80)
    print("✓ All visualizations generated successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. frequency_separation.png - How wavelets separate frequencies")
    print("  2. wavelet_decomposition.png - Wavelet bands (LL, LH, HL, HH)")
    print("  3. triplane_structure.png - 3 orthogonal planes")
    print("  4. pipeline_outputs.png - Complete pipeline visualization")
    print("  5. resolution_comparison.png - Low-res vs high-res comparison")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        main()
    except ImportError:
        print("Error: matplotlib is required for visualization.")
        print("Install with: pip install matplotlib")
