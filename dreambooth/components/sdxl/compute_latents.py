import torch

#   batch = {
#                     "input_ids": input_ids,
#                     "images": pixel_values,
#                     "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
#                 }


def compute_latents_sdxl(batch, accelerator, args, vae, noise_scheduler, weight_dtype, scale_factor):
    pixel_values = batch["images"].to(dtype=weight_dtype)

    with torch.no_grad():
        if args.cache_latents:
            model_input = pixel_values.to(accelerator.device)
        else:
            model_input = vae.encode(
                pixel_values.to(dtype=weight_dtype)
            ).latent_dist.sample()

    # Convert images to latent space
    model_input = model_input * scale_factor

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
    )
    timesteps = timesteps.long()

    # Add noise to the model input according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)


    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    return noisy_model_input, target, timesteps