from dreambooth.shared import models_path

MODEL_DIRS = {
    "Checkpoint": f"{models_path}/Stable-diffusion",
    "Controlnet": f"{models_path}/controlnet",
    "Hypernetwork": f"{models_path}/hypernetworks",
    "LORA": f"{models_path}/Lora",
    "LoCon": f"{models_path}/Lycoris",
    "TextualInversion": f"{models_path}/embeddings",
    "VAE": f"{models_path}/vae"
}

BASE_MODELS = {
    "SD 1.4": "v1x",
    "SD 1.5": "v1x",
    "SD 2.0": "v2x-512",
    "SD 2.0 768": "v2x",
    "SD 2.1": "v2x-512",
    "SD 2.1 768": "v2",
    "SDXL 0.9": "sdxl",
    "SDXL 1.0": "sdxl"
}
