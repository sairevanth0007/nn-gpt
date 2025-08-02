
import itertools
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import argparse
import sys

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, AutoModel
from ab.nn.metric.Clip import create_metric


class Net(nn.Module):
    class TextEncoder(nn.Module):
        def __init__(self, out_size=768):
            super().__init__()
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModel.from_pretrained(model_name)
            self.text_linear = nn.Linear(768, out_size)
            for param in self.text_model.parameters():
                param.requires_grad = False

        def forward(self, text):
            device = self.text_linear.weight.device
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.text_model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device)
            )
            return self.text_linear(outputs.last_hidden_state)

    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.prm = prm or {}

        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.text_encoder = self.TextEncoder(out_size=prm.get('cross_attention_dim', 768)).to(device)

        self.unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(256, 512, 512),
            cross_attention_dim=prm.get('cross_attention_dim', 768)
        ).to(device)

        self.unet.enable_gradient_checkpointing()
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        self.vae.requires_grad_(False)


    @torch.no_grad()
    def forward(self, images, prompts):
        # We don't use the input images for generation, only the prompts.
        # This returns a list of generated PIL images.
        return self.generate(list(prompts))

    def train_setup(self, prm):
        trainable_params = itertools.chain(self.unet.parameters(), self.text_encoder.text_linear.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.prm.get('lr', 1e-5))
        self.criterion = nn.MSELoss()


    def learn(self, train_data):
        self.train()
        total_loss = 0.0

        for batch in train_data:
            images, text_prompts = batch
            self.optimizer.zero_grad()

            with torch.no_grad():
                latents = self.vae.encode(images.to(self.device)).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=self.device)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            text_embeddings = self.text_encoder(text_prompts)

            noise_pred = self.unet(sample=noisy_latents, timestep=timesteps,
                                   encoder_hidden_states=text_embeddings).sample
            loss = self.criterion(noise_pred, noise)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()


        return total_loss / len(train_data) if train_data else 0.0

    @torch.no_grad()
    def generate(self, text_prompts, num_inference_steps=50):
        self.eval()
        text_embeddings = self.text_encoder(text_prompts)
        latents = torch.randn((len(text_prompts), self.unet.config.in_channels, 32, 32), device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(sample=latents, timestep=t, encoder_hidden_states=text_embeddings).sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return [Image.fromarray((img * 255).astype(np.uint8)) for img in images]


def supported_hyperparameters():
    return {'lr', 'momentum'}


if __name__ == "__main__":
    # The standalone inference script remains the same and still works.
    parser = argparse.ArgumentParser(description="Generate images from a text prompt.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt to generate an image for.")
    parser.add_argument("--output_path", type=str, default="generated_image.png",
                        help="Path to save the generated image.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(in_shape=None, out_shape=None, prm={}, device=device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    generated_images = model.generate([args.prompt])
    if generated_images:
        output_image = generated_images[0]
        output_image.save(args.output_path)
        clip_metric = create_metric(device=device)
        clip_metric([output_image], [args.prompt])
        score = clip_metric.result()
        print(f"Image saved to {args.output_path} with a CLIP Score of: {score:.2f}")