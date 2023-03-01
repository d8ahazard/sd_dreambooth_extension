import torch


def get_velocity(
        self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as sample
    self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    timesteps = timesteps.to(sample.device)

    sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(sample.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    return velocity
