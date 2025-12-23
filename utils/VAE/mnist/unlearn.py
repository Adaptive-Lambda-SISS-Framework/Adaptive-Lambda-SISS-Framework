import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
import os
import numpy as np
from tqdm import tqdm
import math


# --- Dataset Loading Functions ---
def load_dataset(pt_file):
    """Load dataset from .pt file"""
    data = torch.load(pt_file, weights_only=False)
    return data["images"], data["labels"]


def create_datasets():
    """Create retain (MNIST) and forget (trousers) datasets"""
    # Load original MNIST (retain set)
    mnist_images, mnist_labels = load_dataset('Dataset/original_mnist.pt')

    # Load trousers subset (forget set)
    trousers_images, trousers_labels = load_dataset('Dataset/trousers_subset.pt')

    return mnist_images, mnist_labels, trousers_images, trousers_labels


# --- VAE Model Definitions (same as before) ---
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


# --- Proper SIS-S Loss for VAE ---
class SISSVAELoss(nn.Module):
    """
    SIS-S (Selective Importance Sampling with Scaling) Loss for VAE
    Proper VAE formulation based on ELBO

    The SIS-S objective for VAE:

    ℒ_SISS = E_{x_r∼R, x_f∼F, z∼q_λ(z|x_r,x_f)} [
        w_r * (log p_θ(x_r|z) - β * D_KL(q_ϕ(z|x_r) || p(z)))
        - (1+s) * w_f * (log p_θ(x_f|z) - β * D_KL(q_ϕ(z|x_f) || p(z)))
    ]

    where:
    - q_λ(z|x_r,x_f) = λ q_ϕ(z|x_r) + (1-λ) q_ϕ(z|x_f) [mixture distribution]
    - w_r = q_ϕ(z|x_r) / q_λ(z|x_r,x_f) [importance weight for retain]
    - w_f = q_ϕ(z|x_f) / q_λ(z|x_r,x_f) [importance weight for forget]
    - β = weight for KL term (usually 1.0)
    - s = scaling parameter for forget term

    For Gaussian VAE with decoder output in [-1, 1]:
    - log p_θ(x|z) ≈ -MSE(x, decoder(z)) [negative reconstruction loss]
    - D_KL(q_ϕ(z|x) || p(z)) = -0.5 * Σ(1 + logσ² - μ² - σ²)
    """

    def __init__(self, lambda_param=0.5, scale_param=1.0, beta_kl=1.0, recon_loss_type="mse"):
        """
        Args:
            lambda_param: λ in mixture distribution (0.5 for equal mixing)
            scale_param: s in the paper (scaling factor for forget term)
            beta_kl: β weight for KL divergence term
            recon_loss_type: "mse" or "bce" for reconstruction loss
        """
        super().__init__()
        self.lambda_param = lambda_param
        self.scale_param = scale_param
        self.beta_kl = beta_kl
        self.recon_loss_type = recon_loss_type

    def gaussian_log_prob(self, z, mu, logvar):
        """
        Compute log probability of z under Gaussian N(mu, exp(logvar))

        log q(z|x) = -0.5 * [ (z-μ)²/σ² + log(2πσ²) ]
                   = -0.5 * [ (z-μ)²/σ² + log(2π) + log(σ²) ]
        """
        # Ensure numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        var = torch.exp(logvar)

        # log q(z|x) = -0.5 * Σ[ (z_i-μ_i)²/σ_i² + log(2π) + log(σ_i²) ]
        log_prob = -0.5 * (
                ((z - mu) ** 2) / var
                + logvar
                + torch.log(torch.tensor(2 * math.pi, device=z.device))
        )

        # Sum over latent dimensions
        return log_prob.sum(dim=1)

    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence: D_KL(q(z|x) || p(z))
        where p(z) = N(0, I)

        KL = -0.5 * Σ(1 + logσ² - μ² - σ²)
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def reconstruction_loss(self, x_recon, x_target):
        """
        Compute negative log likelihood: -log p(x|z)

        For decoder output in [-1, 1] (Tanh activation):
        - If using MSE: -log p(x|z) ∝ MSE(x, decoder(z))
        - If using BCE: treat output as probabilities in [0,1] after scaling
        """
        if self.recon_loss_type == "mse":
            # MSE loss (images normalized to [-1, 1])
            # -log p(x|z) ∝ ||x - decoder(z)||²
            loss = F.mse_loss(x_recon, x_target, reduction='none')
            loss = loss.view(x_target.size(0), -1).mean(dim=1)
            return loss
        elif self.recon_loss_type == "bce":
            # Convert from [-1, 1] to [0, 1] for BCE
            x_target_scaled = (x_target + 1) / 2
            x_recon_scaled = (x_recon + 1) / 2
            # Clamp to avoid log(0)
            x_recon_scaled = torch.clamp(x_recon_scaled, min=1e-7, max=1 - 1e-7)
            # BCE loss
            loss = F.binary_cross_entropy(x_recon_scaled, x_target_scaled, reduction='none')
            loss = loss.view(x_target.size(0), -1).mean(dim=1)
            return loss
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")

    def compute_importance_weights(self, z, mu_r, logvar_r, mu_f, logvar_f):
        """Compute importance weights w_r and w_f"""
        # Compute log probabilities
        log_q_z_given_xr = self.gaussian_log_prob(z, mu_r, logvar_r)
        log_q_z_given_xf = self.gaussian_log_prob(z, mu_f, logvar_f)

        # Compute log of mixture distribution q_λ(z|x_r,x_f)
        # q_λ = λ * q(z|x_r) + (1-λ) * q(z|x_f)
        log_q_lambda = torch.logsumexp(
            torch.stack([
                torch.log(torch.tensor(self.lambda_param, device=z.device)) + log_q_z_given_xr,
                torch.log(torch.tensor(1 - self.lambda_param, device=z.device)) + log_q_z_given_xf
            ]),
            dim=0
        )

        # Compute importance weights (exponentiate carefully)
        w_r = torch.exp(log_q_z_given_xr - log_q_lambda)
        w_f = torch.exp(log_q_z_given_xf - log_q_lambda)

        return w_r, w_f

    def forward(self, x_r, x_f, embed_r, embed_f, vae):
        """
        Compute SIS-S loss for VAE

        Args:
            x_r: Retain images [B, C, H, W]
            x_f: Forget images [B, C, H, W]
            embed_r: Embeddings for retain images [B, embed_dim]
            embed_f: Embeddings for forget images [B, embed_dim]
            vae: VAE model

        Returns:
            total_loss, retain_elbo, forget_elbo, avg_w_r, avg_w_f
        """
        batch_size = x_r.size(0)

        # Get encoder distributions for both sets
        mu_r, logvar_r = vae.encoder(x_r, embed_r)
        mu_f, logvar_f = vae.encoder(x_f, embed_f)

        # Sample from mixture distribution q_λ(z|x_r,x_f)
        # For λ=0.5, we mix retain and forget samples
        if self.lambda_param == 0.5:
            # Equal mixing: sample half from retain, half from forget
            mask = torch.rand(batch_size, device=x_r.device) < 0.5
            z_r = vae.reparameterize(mu_r, logvar_r)
            z_f = vae.reparameterize(mu_f, logvar_f)
            z = torch.where(mask.unsqueeze(-1), z_r, z_f)
        else:
            # General case: sample according to lambda
            mask = torch.rand(batch_size, device=x_r.device) < self.lambda_param
            z_r = vae.reparameterize(mu_r, logvar_r)
            z_f = vae.reparameterize(mu_f, logvar_f)
            z = torch.where(mask.unsqueeze(-1), z_r, z_f)

        # Compute importance weights
        w_r, w_f = self.compute_importance_weights(z, mu_r, logvar_r, mu_f, logvar_f)

        # Decode with both embeddings
        x_rec_r = vae.decoder(z, embed_r)  # Decode as if it came from retain
        x_rec_f = vae.decoder(z, embed_f)  # Decode as if it came from forget

        # Reconstruction losses (negative log likelihood)
        recon_loss_r = self.reconstruction_loss(x_rec_r, x_r)  # -log p(x_r|z)
        recon_loss_f = self.reconstruction_loss(x_rec_f, x_f)  # -log p(x_f|z)

        # KL divergences
        kl_r = self.kl_divergence(mu_r, logvar_r)  # D_KL(q(z|x_r) || p(z))
        kl_f = self.kl_divergence(mu_f, logvar_f)  # D_KL(q(z|x_f) || p(z))

        # ELBO terms (negative since we minimize loss)
        # ELBO = E[log p(x|z)] - β * D_KL
        # But we have -log p(x|z), so: -ELBO = recon_loss + β * KL
        elbo_r = recon_loss_r + self.beta_kl * kl_r
        elbo_f = recon_loss_f + self.beta_kl * kl_f

        # Dataset size ratios (n = total, k = forget)
        total_samples = 60000  # MNIST size
        forget_samples = 1000  # Approximate trousers subset size
        n_over_nk = total_samples / (total_samples - forget_samples)
        k_over_nk = forget_samples / (total_samples - forget_samples)

        # Weighted SIS-S loss
        retain_term = n_over_nk * (w_r * elbo_r).mean()
        forget_term = k_over_nk * (w_f * elbo_f).mean()

        # Total SIS-S loss: maximize ELBO for retain, minimize for forget
        # ℒ_SISS = retain_ELBO - (1+s) * forget_ELBO
        total_loss = retain_term - (1 + self.scale_param) * forget_term

        # Compute average weights for monitoring
        avg_w_r = w_r.mean().item()
        avg_w_f = w_f.mean().item()

        return total_loss, -elbo_r.mean(), -elbo_f.mean(), avg_w_r, avg_w_f


# --- Unlearning Function ---
def unlearn_vae_siss(
        vae_model_path="VAE/vae_final.pt",
        embedder_path="VAE/class_embedder.pt",
        unlearn_epochs=10,
        save_dir="./models/mnist/VAE/Static_SISS_Unlearned/",
        lambda_param=0.5,
        scale_param=1.0,
        learning_rate=1e-5,
        batch_size=32,
        device=None
):
    """
    Unlearn trousers from VAE using SIS-S method (proper VAE formulation)

    Args:
        vae_model_path: Path to trained VAE model
        embedder_path: Path to class embedder
        unlearn_epochs: Number of unlearning epochs
        save_dir: Directory to save unlearned models (won't overwrite original)
        lambda_param: λ parameter for mixture distribution
        scale_param: s parameter for scaling forget term
        learning_rate: Learning rate for unlearning
        batch_size: Batch size for unlearning
        device: Device to use (cuda/cpu)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")
    print(f"Unlearning with λ={lambda_param}, s={scale_param}")
    print("=" * 60)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    retain_images, retain_labels, forget_images, forget_labels = create_datasets()

    print(f"Retain set (MNIST): {len(retain_images)} images")
    print(f"Forget set (trousers): {len(forget_images)} images")

    # Load models
    print("\nLoading models...")

    # Model parameters
    num_classes = 10
    embedding_dim = 64
    latent_dim = 128

    # Initialize models
    class_embedder = nn.Embedding(num_classes, embedding_dim).to(device)
    vae = ConditionalVAE(in_channels=1, embed_dim=embedding_dim, latent_dim=latent_dim).to(device)

    # Load weights
    vae.load_state_dict(torch.load(vae_model_path, map_location=device, weights_only=False))
    class_embedder.load_state_dict(torch.load(embedder_path, map_location=device, weights_only=False))

    print(f"✅ Models loaded from:")
    print(f"   - VAE: {vae_model_path}")
    print(f"   - Embedder: {embedder_path}")

    # Create datasets
    class UnlearningDataset(Dataset):
        def __init__(self, retain_images, retain_labels, forget_images, forget_labels, transform=None):
            self.retain_images = retain_images
            self.retain_labels = retain_labels
            self.forget_images = forget_images
            self.forget_labels = forget_labels
            self.transform = transform
            self.to_pil = ToPILImage()

            # Ensure we have enough forget samples by repeating if needed
            self.forget_len = len(forget_images)
            self.retain_len = len(retain_images)

        def __len__(self):
            return max(self.retain_len, self.forget_len)

        def __getitem__(self, idx):
            # Sample retain and forget pairs
            retain_idx = idx % self.retain_len
            forget_idx = idx % self.forget_len

            x_r = self.retain_images[retain_idx]
            label_r = self.retain_labels[retain_idx]
            x_f = self.forget_images[forget_idx]
            label_f = self.forget_labels[forget_idx]

            # Convert to PIL and apply transform if needed
            if x_r.ndim == 2 or (x_r.ndim == 3 and x_r.shape[0] == 1):
                x_r = x_r.unsqueeze(0) if x_r.ndim == 2 else x_r
                x_r = self.to_pil(x_r.squeeze(0))

            if x_f.ndim == 2 or (x_f.ndim == 3 and x_f.shape[0] == 1):
                x_f = x_f.unsqueeze(0) if x_f.ndim == 2 else x_f
                x_f = self.to_pil(x_f.squeeze(0))

            # Apply transform
            if self.transform:
                x_r = self.transform(x_r)
                x_f = self.transform(x_f)

            # Ensure correct shape
            if x_r.ndim == 2:
                x_r = x_r.unsqueeze(0)
            if x_f.ndim == 2:
                x_f = x_f.unsqueeze(0)

            return x_r, label_r, x_f, label_f

    # Transform for unlearning (same as training)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(p=0.5),
        ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = UnlearningDataset(
        retain_images, retain_labels,
        forget_images, forget_labels,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues on Windows
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize loss and optimizer
    criterion = SISSVAELoss(
        lambda_param=lambda_param,
        scale_param=scale_param,
        beta_kl=1.0,
        recon_loss_type="mse"  # Use MSE for [-1, 1] normalized images
    )

    optimizer = torch.optim.AdamW(
        list(vae.parameters()) + list(class_embedder.parameters()),
        lr=learning_rate,
        weight_decay=1e-6
    )

    # Scheduler for learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=unlearn_epochs * len(dataloader)
    )

    # Training loop
    print(f"\nStarting unlearning for {unlearn_epochs} epochs...")
    print("=" * 60)

    vae.train()
    class_embedder.train()

    for epoch in range(unlearn_epochs):
        total_loss = 0.0
        total_retain_elbo = 0.0  # Positive ELBO (higher is better)
        total_forget_elbo = 0.0  # Positive ELBO (we want to minimize this)
        total_w_r = 0.0
        total_w_f = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{unlearn_epochs}")
        for batch_idx, (x_r_batch, label_r_batch, x_f_batch, label_f_batch) in enumerate(pbar):
            # Move to device
            x_r_batch = x_r_batch.to(device).float()
            x_f_batch = x_f_batch.to(device).float()
            label_r_batch = label_r_batch.to(device)
            label_f_batch = label_f_batch.to(device)

            # Get embeddings
            embed_r = class_embedder(label_r_batch)
            embed_f = class_embedder(label_f_batch)

            # Forward pass and loss computation
            optimizer.zero_grad()

            loss, retain_elbo, forget_elbo, avg_w_r, avg_w_f = criterion(
                x_r_batch, x_f_batch, embed_r, embed_f, vae
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(vae.parameters()) + list(class_embedder.parameters()),
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            # Update metrics
            total_loss += loss.item()
            total_retain_elbo += retain_elbo.item()
            total_forget_elbo += forget_elbo.item()
            total_w_r += avg_w_r
            total_w_f += avg_w_f
            num_batches += 1

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Retain_ELBO': f'{retain_elbo.item():.4f}',
                'Forget_ELBO': f'{forget_elbo.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })

        # Epoch statistics
        avg_loss = total_loss / num_batches
        avg_retain_elbo = total_retain_elbo / num_batches
        avg_forget_elbo = total_forget_elbo / num_batches
        avg_w_r = total_w_r / num_batches
        avg_w_f = total_w_f / num_batches

        print(f"\nEpoch {epoch + 1}/{unlearn_epochs}:")
        print(f"  Total SIS-S Loss: {avg_loss:.6f}")
        print(f"  Retain ELBO: {avg_retain_elbo:.6f} (higher = better retention)")
        print(f"  Forget ELBO: {avg_forget_elbo:.6f} (lower = better forgetting)")
        print(f"  Avg weight w_r: {avg_w_r:.4f}, w_f: {avg_w_f:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 40)

        # Save checkpoint every few epochs
        if (epoch + 1) % 10 == 0 or epoch == unlearn_epochs - 1:
            checkpoint_path = f"{save_dir}/vae_unlearned_epoch_{epoch + 1:03d}.pt"
            embedder_checkpoint_path = f"{save_dir}/embedder_unlearned_epoch_{epoch + 1:03d}.pt"

            torch.save(vae.state_dict(), checkpoint_path)
            torch.save(class_embedder.state_dict(), embedder_checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")

    # Save final static-SISS unlearned models
    final_vae_path = f"{save_dir}/vae_static_siss_unlearned_final.pt"
    final_embedder_path = f"{save_dir}/static_siss_embedder_unlearned_final.pt"

    torch.save(vae.state_dict(), final_vae_path)
    torch.save(class_embedder.state_dict(), final_embedder_path)

    print("\n" + "=" * 60)
    print("✅ Unlearning completed!")
    print(f"Original models preserved at:")
    print(f"  - {vae_model_path}")
    print(f"  - {embedder_path}")
    print(f"\nUnlearned models saved to:")
    print(f"  - {final_vae_path}")
    print(f"  - {final_embedder_path}")
    print("=" * 60)

    return vae, class_embedder


# --- Quick Testing Function ---
def test_unlearning(vae, embedder, device, class_label=1, num_samples=5):
    """Quick test to see if unlearning worked"""
    print(f"\nTesting generation for class {class_label}...")

    vae.eval()
    embedder.eval()

    with torch.no_grad():
        # Create embeddings for the class
        labels = torch.full((num_samples,), class_label, device=device)
        embeddings = embedder(labels)

        # Sample from prior
        z = torch.randn(num_samples, vae.decoder.latent_dim, device=device)

        # Generate images
        generated = vae.decoder(z, embeddings)
        generated = (generated.clamp(-1, 1) + 1) / 2.0  # to [0,1]

        # Check variance (should be high if model is uncertain about trousers)
        variance = generated.view(num_samples, -1).var(dim=0).mean().item()
        print(f"Average pixel variance: {variance:.6f}")
        print("(Higher variance suggests better unlearning of trousers)")

    vae.train()
    embedder.train()


# --- Main execution ---
if __name__ == "__main__":
    # Example usage
    print("VAE Unlearning using SIS-S Method (Proper VAE Formulation)")
    print("=" * 60)

    print("\nTheory:")
    print("-" * 40)
    print("SIS-S for VAE objective:")
    print("ℒ_SISS = E_{x_r∼R, x_f∼F, z∼q_λ(z|x_r,x_f)} [")
    print("  w_r * (log p(x_r|z) - β D_KL(q(z|x_r) || p(z)))")
    print("  - (1+s) * w_f * (log p(x_f|z) - β D_KL(q(z|x_f) || p(z)))")
    print("]")
    print("\nwhere:")
    print("- q_λ(z|x_r,x_f) = λ q(z|x_r) + (1-λ) q(z|x_f)")
    print("- w_r = q(z|x_r) / q_λ(z|x_r,x_f)")
    print("- w_f = q(z|x_f) / q_λ(z|x_r,x_f)")
    print("- β = KL weight (1.0)")
    print("- s = scaling parameter for forget term")
    print("=" * 60)

    # Unlearn trousers from the model
    unlearned_vae, unlearned_embedder = unlearn_vae_siss(
        vae_model_path="./utils/VAE/mnist/vae_final.pt",
        embedder_path="./utils/VAE/mnist/class_embedder.pt",
        unlearn_epochs=20,  # Number of unlearning epochs
        save_dir="VAE_unlearned",  # Won't overwrite original files
        lambda_param=0.5,  # λ = 1/2 as specified in paper
        scale_param=1.0,  # s = 1 for standard unlearning
        learning_rate=1e-5,  # Lower LR for fine-tuning
        batch_size=16,  # Smaller batch for stability
        device="cuda"  # Use GPU if available
    )

    # Quick test
    test_unlearning(unlearned_vae, unlearned_embedder, device=torch.device("cuda"))

    print("\nTo test the unlearned model with sampling code:")
    print("python sample_vae.py \\")
    print("  --model_path VAE_unlearned/vae_unlearned_final.pt \\")
    print("  --embedder_path VAE_unlearned/embedder_unlearned_final.pt \\")
    print("  --mode class --class_idx 1 --samples_per_class 10")