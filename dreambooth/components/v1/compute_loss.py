import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_snr


def compute_v1_loss(with_prior, accelerator, unet, noise_scheduler, args, noisy_model_input, timesteps, target, prior_loss, encoder_hidden_states):
    with_prior_preservation = with_prior and args.prior_loss_weight > 0
    class_labels = None
    with accelerator.autocast():
        model_pred = unet(
            noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels
        ).sample

    if model_pred.shape[1] == 6:
        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

    if with_prior_preservation:
        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)
        # Compute prior loss
        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

    # Compute instance loss
    if args.min_snr_gamma == 0:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        base_weight = (
                torch.stack([snr, args.min_snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )

        if noise_scheduler.config.prediction_type == "v_prediction":
            # Velocity objective needs to be floored to an SNR weight of one.
            mse_loss_weights = base_weight + 1
        else:
            # Epsilon and sample both use the same loss weights.
            mse_loss_weights = base_weight
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

    if with_prior_preservation:
        # Add the prior loss to the instance loss.
        loss = loss + args.prior_loss_weight * prior_loss
    return loss, prior_loss