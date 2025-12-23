import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import os
from pathlib import Path
import matplotlib.pyplot as plt


# --- Model Definitions ---
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=0, latent_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        input_channels = in_channels + (embed_dim if embed_dim > 0 else 0)

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.flat_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x, embed=None):
        if self.embed_dim and embed is not None:
            b = x.size(0)
            e = embed.view(b, self.embed_dim, 1, 1).expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, e], dim=1)

        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, out_channels=1, embed_dim=0, latent_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        in_dim = latent_dim + (embed_dim if embed_dim > 0 else 0)

        self.fc = nn.Linear(in_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, embed=None):
        if self.embed_dim and embed is not None:
            z = torch.cat([z, embed], dim=1)
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        xrec = self.deconv(h)
        return xrec


class ConditionalVAE(nn.Module):
    def __init__(self, in_channels=1, embed_dim=0, latent_dim=128):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, embed_dim=embed_dim, latent_dim=latent_dim)
        self.decoder = ConvDecoder(out_channels=in_channels, embed_dim=embed_dim, latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, embed=None):
        mu, logvar = self.encoder(x, embed)
        z = self.reparameterize(mu, logvar)
        xrec = self.decoder(z, embed)
        return xrec, mu, logvar


# --- Model Loading Function ---
def load_vae_models(model_path="VAE/vae_final.pt", embedder_path="VAE/class_embedder.pt", device=None):
    """
    Load trained VAE and class embedder models.

    Args:
        model_path: Path to VAE model checkpoint
        embedder_path: Path to class embedder checkpoint
        device: Device to load models on (cuda/cpu). If None, auto-detects.

    Returns:
        vae: Loaded VAE model
        embedder: Loaded class embedder
        device: Device used
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Loading models on {device}...")

    # Model parameters (should match training)
    num_classes = 10
    embedding_dim = 64
    latent_dim = 128

    # Initialize models
    class_embedder = nn.Embedding(num_classes, embedding_dim).to(device)
    vae = ConditionalVAE(in_channels=1, embed_dim=embedding_dim, latent_dim=latent_dim).to(device)

    # Load weights
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    class_embedder.load_state_dict(torch.load(embedder_path, map_location=device, weights_only=False))

    # Set to eval mode
    vae.eval()
    class_embedder.eval()

    print(f"✅ Models loaded successfully!")
    print(f"   - VAE from: {model_path}")
    print(f"   - Embedder from: {embedder_path}")

    return vae, class_embedder, device


# --- Main Sampling Functions ---
def sample_single_class(vae, embedder, device, class_label=0, num_samples=10,
                        output_dir="VAE/samples", return_images=False):
    """
    Generate samples for a single class.

    Args:
        vae: Loaded VAE model
        embedder: Loaded class embedder
        device: Device to use
        class_label: Class label to generate (0-9)
        num_samples: Number of samples to generate
        output_dir: Directory to save images
        return_images: If True, return generated images as tensor

    Returns:
        If return_images=True: tensor of generated images (num_samples, 1, 64, 64)
        Otherwise: None
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Get class embedding
        label_tensor = torch.full((num_samples,), class_label, device=device)
        embeddings = embedder(label_tensor)

        # Sample z from normal distribution
        z = torch.randn(num_samples, vae.decoder.latent_dim, device=device)

        # Generate images
        recon = vae.decoder(z, embeddings)
        recon = (recon.clamp(-1, 1) + 1) / 2.0  # Convert to [0,1]

    # Create grid of images
    grid_nrow = min(5, num_samples)
    grid = make_grid(recon, nrow=grid_nrow)

    # Save grid image
    grid_filename = f"{output_dir}/class_{class_label}_grid.png"
    save_image(grid, grid_filename)
    print(f"✅ Saved grid image: {grid_filename}")

    # Save individual images
    for i in range(num_samples):
        img_filename = f"{output_dir}/class_{class_label}_sample_{i:03d}.png"
        save_image(recon[i], img_filename)

    print(f"✅ Generated {num_samples} samples for class {class_label} in '{output_dir}'")

    if return_images:
        return recon


def sample_all_classes(vae, embedder, device, samples_per_class=5,
                       output_dir="VAE/all_classes", return_images=False):
    """
    Generate samples for all classes.

    Args:
        vae: Loaded VAE model
        embedder: Loaded class embedder
        device: Device to use
        samples_per_class: Number of samples per class
        output_dir: Directory to save images
        return_images: If True, return generated images as tensor

    Returns:
        If return_images=True: tuple of (images tensor, labels tensor)
        Otherwise: None
    """
    n_classes = 10
    total_samples = n_classes * samples_per_class

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Prepare labels and embeddings for all classes
        labels = torch.arange(n_classes, device=device)
        embeddings = embedder(labels)  # (num_classes, embedding_dim)

        # Create repeat for number of samples per class
        embeddings = embeddings.unsqueeze(1).repeat(1, samples_per_class, 1)
        embeddings = embeddings.view(-1, embeddings.size(-1))

        # Create corresponding labels
        all_labels = labels.unsqueeze(1).repeat(1, samples_per_class).view(-1)

        # Sample z from normal distribution
        z = torch.randn(total_samples, vae.decoder.latent_dim, device=device)

        # Generate images
        recon = vae.decoder(z, embeddings)
        recon = (recon.clamp(-1, 1) + 1) / 2.0  # Convert to [0,1]

    # Save overall grid
    overall_grid = make_grid(recon, nrow=samples_per_class)
    save_image(overall_grid, f"{output_dir}/all_classes_grid.png")
    print(f"✅ Saved overall grid: {output_dir}/all_classes_grid.png")

    # Save per-class grids
    for cls in range(n_classes):
        start_idx = cls * samples_per_class
        end_idx = (cls + 1) * samples_per_class
        class_grid = make_grid(recon[start_idx:end_idx], nrow=samples_per_class)
        save_image(class_grid, f"{output_dir}/class_{cls}_grid.png")

    # Save individual images
    for i in range(total_samples):
        cls = all_labels[i].item()
        save_image(recon[i], f"{output_dir}/class_{cls}_sample_{i % samples_per_class:02d}.png")

    print(f"✅ Generated {total_samples} samples (all classes) in '{output_dir}'")

    if return_images:
        return recon, all_labels


def interpolate_class(vae, embedder, device, class_label=0, num_points=7,
                      output_dir="VAE/interpolation", return_images=False):
    """
    Create interpolation in latent space for a specific class.

    Args:
        vae: Loaded VAE model
        embedder: Loaded class embedder
        device: Device to use
        class_label: Class label to interpolate (0-9)
        num_points: Number of interpolation points
        output_dir: Directory to save images
        return_images: If True, return interpolated images as tensor

    Returns:
        If return_images=True: tensor of interpolated images
        Otherwise: None
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get class embedding
    label_tensor = torch.tensor([class_label], device=device)
    embed = embedder(label_tensor)

    # Create two random latent vectors
    z1 = torch.randn(1, vae.decoder.latent_dim, device=device)
    z2 = torch.randn(1, vae.decoder.latent_dim, device=device)

    # Interpolate between them
    interpolated_images = []
    with torch.no_grad():
        for alpha in torch.linspace(0, 1, num_points):
            z = (1 - alpha) * z1 + alpha * z2
            recon = vae.decoder(z, embed)
            recon = (recon.clamp(-1, 1) + 1) / 2.0  # Convert to [0,1]
            interpolated_images.append(recon)

    # Concatenate all images
    all_images = torch.cat(interpolated_images, dim=0)

    # Save interpolation grid
    grid = make_grid(all_images, nrow=num_points)
    grid_filename = f"{output_dir}/interpolation_class_{class_label}.png"
    save_image(grid, grid_filename)
    print(f"✅ Saved interpolation: {grid_filename}")

    # Save individual interpolation points
    for i, alpha in enumerate(torch.linspace(0, 1, num_points)):
        save_image(all_images[i], f"{output_dir}/interp_class_{class_label}_alpha_{alpha:.2f}.png")

    print(f"✅ Generated {num_points} interpolation points for class {class_label} in '{output_dir}'")

    if return_images:
        return all_images


# --- Convenience Function ---
def quick_sample(class_label=0, num_samples=10,
                 model_path="./VAE/Baseline/vae_final.pt",
                 embedder_path="./VAE/Baseline/class_embedder.pt",
                 output_dir="./VAE/Samples/",
                 device=None):
    """
    Quick sampling function - loads models and generates samples in one call.

    Args:
        class_label: Class label to generate (0-9)
        num_samples: Number of samples to generate
        model_path: Path to VAE model
        embedder_path: Path to class embedder
        output_dir: Directory to save images
        device: Device to use (cuda/cpu), None for auto

    Returns:
        Generated images tensor
    """
    print(f"🔧 Quick sampling for class {class_label}...")

    # Load models
    vae, embedder, device = load_vae_models(model_path, embedder_path, device)

    # Generate samples
    images = sample_single_class(
        vae, embedder, device,
        class_label=class_label,
        num_samples=num_samples,
        output_dir=output_dir,
        return_images=True
    )

    print(f"✅ Quick sampling completed!")
    return images


# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Quick sampling for class 3
    print("=" * 50)
    print("Example 1: Quick sampling for class 3")
    print("=" * 50)
    images_class3 = quick_sample(
        class_label=3,
        num_samples=500,
        model_path="./VAE/Baseline/vae_final.pt",
        embedder_path="./VAE/Baseline/class_embedder.pt",
        output_dir="./VAE/Samples/Baseline/class_3"
    )

    # # Example 2: Load models once and sample multiple times
    # print("\n" + "=" * 50)
    # print("Example 2: Load models and sample multiple classes")
    # print("=" * 50)

    # # Load models
    # vae, embedder, device = load_vae_models(
    #     model_path="VAE/vae_final.pt",
    #     embedder_path="VAE/class_embedder.pt"
    # )

    # # Sample class 5
    # images_class5 = sample_single_class(
    #     vae, embedder, device,
    #     class_label=5,
    #     num_samples=8,
    #     output_dir="VAE/class_5_samples",
    #     return_images=True
    # )

    # # Sample all classes
    # all_images, all_labels = sample_all_classes(
    #     vae, embedder, device,
    #     samples_per_class=3,
    #     output_dir="VAE/all_classes_samples",
    #     return_images=True
    # )

    # # Create interpolation for class 7
    # interp_images = interpolate_class(
    #     vae, embedder, device,
    #     class_label=7,
    #     num_points=9,
    #     output_dir="VAE/interpolation_class_7",
    #     return_images=True
    # )

    # print("\n" + "=" * 50)
    # print("✅ All sampling examples completed!")
    # print("=" * 50)