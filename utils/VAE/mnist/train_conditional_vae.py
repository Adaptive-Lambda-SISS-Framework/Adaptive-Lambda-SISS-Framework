import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
# Remove the old imports
# from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import ToPILImage, ToTensor

# --- Dataset Loading ---
pt_file = "./data/augmented_mnist_with_trousers.pt"
data = torch.load(pt_file, weights_only=False)
images = data["images"]
labels = data["labels"]

# --- Dataset and Transform ---
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(p=0.5),
    ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        x = self.to_pil(x)
        if self.transform:
            x = self.transform(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x, self.labels[idx]


# --- Main execution block ---
def main():
    # --- Dataloader ---
    batch_size = 128
    dataset = CustomDataset(images, labels, transform)

    # Use 0 workers on Windows to avoid multiprocessing issues
    # You can try increasing this on Linux/Mac or if you fix the Windows issue
    num_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Conditional VAE Model ---
    num_classes = 10
    embedding_dim = 64  # smaller than before; used for conditioning
    latent_dim = 128

    class ConvEncoder(nn.Module):
        def __init__(self, in_channels=1, embed_dim=0, latent_dim=128):
            super().__init__()
            self.embed_dim = embed_dim
            input_channels = in_channels + (embed_dim if embed_dim > 0 else 0)

            # conv stack -> feature map -> flatten -> fc to mu/logvar
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, 4, 2, 1),  # 64 -> 32
                nn.BatchNorm2d(32),
                nn.ReLU(True),

                nn.Conv2d(32, 64, 4, 2, 1),  # 32 -> 16
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                nn.Conv2d(64, 128, 4, 2, 1),  # 16 -> 8
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                nn.Conv2d(128, 256, 4, 2, 1),  # 8 -> 4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )

            self.flat_dim = 256 * 4 * 4
            self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
            self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        def forward(self, x, embed=None):
            # x: (B, 1, 64, 64)
            if self.embed_dim and embed is not None:
                # embed: (B, embed_dim) -> expand to (B, embed_dim, 64,64)
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

            # project to feature map 256 x 4 x 4
            self.fc = nn.Linear(in_dim, 256 * 4 * 4)

            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 -> 8
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 -> 32
                nn.BatchNorm2d(32),
                nn.ReLU(True),

                nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # 32 -> 64
                nn.Tanh()  # outputs in [-1,1] consistent with normalize transform
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

    # --- Instantiate models and class embedding ---
    class_embedder = nn.Embedding(num_classes, embedding_dim).to(device)
    vae = ConditionalVAE(in_channels=1, embed_dim=embedding_dim, latent_dim=latent_dim).to(device)

    # --- Checkpoint loading (optional) ---
    os.makedirs("VAE", exist_ok=True)
    model_ckpt = "./VAE/vae_final.pt"
    embedder_ckpt = "./VAE/class_embedder.pt"

    if os.path.exists(model_ckpt) and os.path.exists(embedder_ckpt):
        print(f"✅ Found pretrained VAE at {model_ckpt}. Loading weights...")
        vae.load_state_dict(torch.load(model_ckpt, map_location=device, weights_only=False))
        class_embedder.load_state_dict(torch.load(embedder_ckpt, map_location=device, weights_only=False))
    else:
        print("⚠️ No pretrained VAE found. Starting training from scratch.")

    # --- Optimizer, LR scheduler, AMP scaler ---
    optimizer = torch.optim.AdamW(
        list(vae.parameters()) + list(class_embedder.parameters()),
        lr=2e-4,
        weight_decay=1e-6
    )
    use_amp = device.type == "cuda"
    # Fixed GradScaler - use new syntax
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    epochs = 50
    total_steps = len(dataloader) * epochs
    # Simple step LR or cosine warm restarts is fine — keep simple
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Loss function ---
    def vae_loss(recon_x, x, mu, logvar, beta=1.0):
        # recon loss: MSE (images normalized to [-1,1]); you can use BCE if images in [0,1]
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        # KL divergence between q(z|x) ~ N(mu, sigma) and p(z) ~ N(0, I)
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl, recon_loss, kl

    # --- Sampling Function (conditional) ---
    def sample_images(epoch, vae_model, embedder, device, n_classes=10, samples_per_class=1, out_dir="VAE/samples"):
        print(f"\nSampling images at epoch {epoch}...")
        vae_model.eval()
        embedder.eval()
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            labels = torch.arange(n_classes, device=device)
            # prepare embeddings sized (n_classes, embedding_dim)
            embeddings = embedder(labels)  # (num_classes, embedding_dim)
            # create repeat for number of samples per class
            embeddings = embeddings.unsqueeze(1).repeat(1, samples_per_class, 1).view(-1, embeddings.size(-1))
            # sample z
            z = torch.randn(n_classes * samples_per_class, latent_dim, device=device)
            recon = vae_model.decoder(z, embeddings)  # (B, 1, 64,64)
            recon = (recon.clamp(-1, 1) + 1) / 2.0  # to [0,1]

            # Create a grid of images and save it
            # Each class in its own row
            recon_grid = recon.reshape(n_classes, samples_per_class, 1, 64, 64)
            recon_grid = recon_grid.view(n_classes * samples_per_class, 1, 64, 64)

            # Save individual images
            for i in range(recon.size(0)):
                cls = i // samples_per_class
                save_image(
                    recon[i],
                    f"{out_dir}/sample_epoch_{epoch:03d}_class_{cls}_idx_{i % samples_per_class}.png"
                )

            # Also save a grid image
            grid_filename = f"{out_dir}/grid_epoch_{epoch:03d}.png"
            save_image(recon_grid, grid_filename, nrow=samples_per_class)

        print(f"Saved samples to {out_dir}")
        vae_model.train()
        embedder.train()

    # --- Training Loop ---
    vae.train()
    class_embedder.train()
    global_step = 0
    log_interval = 50

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_loss = 0.0

        for images_batch, labels_batch in tqdm(dataloader):
            images_batch = images_batch.to(device).float()
            labels_batch = labels_batch.to(device)

            embeddings = class_embedder(labels_batch)  # (B, embedding_dim)

            optimizer.zero_grad()

            # Use autocast only if AMP is enabled - use the new torch.amp.autocast syntax
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    recon_batch, mu, logvar = vae(images_batch, embeddings)
                    loss, recon_l, kl_l = vae_loss(recon_batch, images_batch, mu, logvar, beta=1.0)
            else:
                recon_batch, mu, logvar = vae(images_batch, embeddings)
                loss, recon_l, kl_l = vae_loss(recon_batch, images_batch, mu, logvar, beta=1.0)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_recon += recon_l.item() * images_batch.size(0)
            epoch_kl += kl_l.item() * images_batch.size(0)
            epoch_loss += loss.item() * images_batch.size(0)
            global_step += 1

            if global_step % log_interval == 0:
                print(f"Step {global_step}: loss={loss.item():.6f}, recon={recon_l.item():.6f}, kl={kl_l.item():.6f}")

        # scheduler step per epoch
        lr_scheduler.step()

        n_samples = len(dataset)
        avg_recon = epoch_recon / n_samples
        avg_kl = epoch_kl / n_samples
        avg_loss = epoch_loss / n_samples
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f} (Recon: {avg_recon:.6f}, KL: {avg_kl:.6f})")

        # Sampling every 5 epochs and last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            sample_images(epoch + 1, vae, class_embedder, device, n_classes=num_classes, samples_per_class=1)

        # Save checkpoint each epoch (keeps latest)
        if (epoch % 10) == 0:
            ckpt_path = f"./utils/VAE/Baseline/vae_epoch_{epoch + 1:03d}.pt"
            emb_path = f"./utils/VAE/Baseline/class_embedder_epoch_{epoch + 1:03d}.pt"
            torch.save(vae.state_dict(), ckpt_path)
            torch.save(class_embedder.state_dict(), emb_path)

    # --- Final Save ---
    final_model_path = "./utils/VAE/Baseline/vae_final.pt"
    final_embed_path = "./utils/VAE/Baseline/class_embedder.pt"
    torch.save(vae.state_dict(), final_model_path)
    torch.save(class_embedder.state_dict(), final_embed_path)
    print(f"Training complete. Final model saved to {final_model_path} and {final_embed_path}.")


if __name__ == '__main__':
    main()