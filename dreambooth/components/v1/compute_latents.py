import torch

from dreambooth.utils.text_utils import encode_hidden_state


def compute_latents_v1(batch, accelerator, args, text_encoder, vae, unet, noise_scheduler, weight_dtype, train_tenc, tokenizer_max_length, scale_factor):
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

    # Sample noise that we'll add to the model input
    if args.offset_noise:
        noise = torch.randn_like(model_input) + 0.1 * torch.randn(
            model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
        )
    else:
        noise = torch.randn_like(model_input)
    bsz, channels, height, width = model_input.shape
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
    )
    timesteps = timesteps.long()

    # Add noise to the model input according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

    pad_tokens = args.pad_tokens if train_tenc else False
    input_ids = batch["input_ids"]

    # Get the text embedding for conditioning
    encoder_hidden_states = None
    if text_encoder is not None:
        encoder_hidden_states = encode_hidden_state(
            text_encoder,
            input_ids,
            pad_tokens,
            bsz,
            args.max_token_length,
            tokenizer_max_length,
            args.clip_skip,
        )

    if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

    if args.class_labels_conditioning == "timesteps":
        class_labels = timesteps
    else:
        class_labels = None

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    return noisy_model_input, target, timesteps, encoder_hidden_states, class_labels
