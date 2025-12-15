import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
import glob
import re


def sample(model_name: str, index: int):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model paths selection
    match model_name:
        case "vanilla":
            unet_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/unet_final.pt"
            cl_embedder_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/class_embedder.pt"
        case "ssis":
            unet_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned/unet_unlearned.pt"
            cl_embedder_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned/class_embedder_unlearned.pt"
        case "dynamic":
            unet_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned_dynamic/unet_unlearned_vae.pt"
            cl_embedder_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned_dynamic/class_embedder_unlearned_vae.pt"
        case _:
            raise ValueError(f"Unknown model_name: {model_name}")

    # Load model
    model = UNet2DConditionModel(
        sample_size=64,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=256,
    ).to(device)
    if not os.path.isfile(unet_path):
        raise FileNotFoundError(f"UNet checkpoint not found at {unet_path}")
    model.load_state_dict(torch.load(unet_path, map_location=device))
    model.eval()

    # Load class embedder
    class_embedder = nn.Embedding(10, 256).to(device)
    if not os.path.isfile(cl_embedder_path):
        raise FileNotFoundError(f"Class embedder checkpoint not found at {cl_embedder_path}")
    class_embedder.load_state_dict(torch.load(cl_embedder_path, map_location=device))
    class_embedder.eval()

    # Noise scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

    # -------- User input for conditional generation --------
    condition_class = index  # digit 0–9
    batch_size = 64
    labels = torch.full((batch_size,), condition_class, dtype=torch.long, device=device)

    # Start from noise
    samples = torch.randn(batch_size, 1, 64, 64, device=device)

    # Sampling loop
    with torch.no_grad():
        for t in tqdm(reversed(range(scheduler.config.num_train_timesteps)),
                      desc=f"Sampling for class {condition_class}"):
            # t is int timestep
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            embeddings = class_embedder(labels).unsqueeze(1)  # (batch_size, 1, 256)

            # Predict noise
            noise_pred = model(samples, t_tensor, encoder_hidden_states=embeddings).sample

            new_samples = []
            for i in range(batch_size):
                step_result = scheduler.step(
                    noise_pred[i].unsqueeze(0),  # model output for sample i
                    t,  # current timestep as int
                    samples[i].unsqueeze(0)  # current sample
                )
                # Depending on diffusers version, prev_sample might be attribute or direct return
                prev = getattr(step_result, "prev_sample", None)
                if prev is None:
                    # older/newer API fallback if step_result is tensor
                    prev = step_result
                new_samples.append(prev)
            samples = torch.cat(new_samples, dim=0)

    # Post-processing
    samples = samples.clamp(0, 1)
    samples_resized = F.interpolate(samples, size=(28, 28), mode='bilinear', align_corners=False)

    # Determine save paths
    if model_name == "vanilla":
        output_path = f"/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Samples/class {condition_class}"
        base_name = f"sample_class{condition_class}"
    elif model_name == "ssis":
        output_path = f"/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Samples/class {condition_class}-Unlearned with constant lambda"
        base_name = f"sample_class{condition_class}"
    elif model_name == "dynamic":
        output_path = f"/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Samples/class {condition_class}-Unlearned with dynamic lambda"
        base_name = f"sample_class{condition_class}"
    else:
        raise ValueError(f"Unhandled model_name: {model_name}")

    os.makedirs(output_path, exist_ok=True)

    # Find existing files to avoid overwrite
    pattern = os.path.join(output_path, f"{base_name}_*.png")
    existing_files = glob.glob(pattern)
    max_idx = -1
    regex = re.compile(rf"{re.escape(base_name)}_(\d+)\.png$")
    for f in existing_files:
        m = regex.search(os.path.basename(f))
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    start_idx = max_idx + 1  # next available index (if no files, starts at 0)

    for offset, img in enumerate(samples_resized.cpu()):
        idx_to_save = start_idx + offset
        filename = f"{base_name}_{idx_to_save}.png"
        save_image(img, os.path.join(output_path, filename))

    ## Plot the images grid
    # grid_img = make_grid(samples_resized.cpu(), nrow=8, normalize=True, pad_value=1)
    # plt.figure(figsize=(12, 12))
    # plt.axis("off")
    # plt.title(f"Generated Samples for Class {condition_class}")
    # plt.imshow(grid_img.permute(1, 2, 0).squeeze(), cmap="gray")
    # plt.show()




if __name__ == "__main__":
    sample(model_name="vanilla", index=1)
    # for j in range(1,2):
    #     print(f"{j }-----------------------------------")
    #     for i in range(1,10):
    #         sample(model_name="siss", index=i)