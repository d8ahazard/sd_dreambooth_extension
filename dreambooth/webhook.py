import hashlib
import os
from enum import Enum
from typing import Union, List

import discord_webhook
from PIL import Image

from dreambooth import shared
from dreambooth.utils.image_utils import image_grid


class DreamboothWebhookTarget(Enum):
    UNKNOWN = 1
    DISCORD = 2


db_path = os.path.join(shared.models_path, "dreambooth")
url_file = os.path.join(db_path, "webhook.txt")
hook_url = None

if not os.path.exists(db_path):
    os.makedirs(db_path)


# Yes, this is absolutely a duplicate of get_secret...
def get_webhook_url():
    url = ""
    if not os.path.exists(url_file):
        return url
    with open(url_file, 'r') as file:
        url = file.read().replace('\n', '')
    return url


def save_and_test_webhook(url: str) -> str:
    global hook_url
    if len(url) <= 0:
        return "Invalid webhook url."
    with open(url_file, 'w') as file:
        file.write(url)
    hook_url = url
    target = __detect_webhook_target(url)
    if target == DreamboothWebhookTarget.DISCORD:
        return __test_discord(url)

    return "Unsupported target."


hook_url = get_webhook_url()


def send_training_update(
        imgs: Union[List[str], str],
        model_name: str,
        prompt: Union[List[str], str],
        training_step: Union[str, int],
        global_step: Union[str, int],
):
    global hook_url
    target = __detect_webhook_target(hook_url)
    if target == DreamboothWebhookTarget.UNKNOWN:
        return  # early return

    # Accept a list, make a grid
    if isinstance(imgs, List):
        out_imgs = [Image.open(img) for img in imgs]

        image = image_grid(out_imgs)

        for i in out_imgs:
            i.close()

        del out_imgs
    else:
        image = Image.open(imgs)

    if isinstance(prompt, List):
        _prompts = prompt
        prompt = ""

        for i, p in enumerate(_prompts, start=1):
            prompt += f"{i}: {p}\r\n"

    if target == DreamboothWebhookTarget.DISCORD:
        __send_discord_training_update(hook_url, image, model_name, prompt.strip(), training_step, global_step)

    image.close()


def _is_valid_notification_target(url: str) -> bool:
    return __detect_webhook_target(url) != DreamboothWebhookTarget.UNKNOWN


def __detect_webhook_target(url: str) -> DreamboothWebhookTarget:
    if url.startswith("https://discord.com/api/webhooks/"):
        return DreamboothWebhookTarget.DISCORD
    return DreamboothWebhookTarget.UNKNOWN


def __test_discord(url: str) -> str:
    discord = discord_webhook.DiscordWebhook(url, username="Dreambooth")
    discord.set_content("This is a test message from the A1111 Dreambooth Extension.")

    response = discord.execute()
    if response.ok:
        return "Test successful."

    return "Test failed."


def __send_discord_training_update(
        url: str,
        image,
        model_name: str,
        prompt: str,
        training_step: Union[str, int],
        global_step: Union[str, int],
):
    discord = discord_webhook.DiscordWebhook(url, username="Dreambooth")
    out = discord_webhook.DiscordEmbed(color="C70039")

    out.set_author(name=model_name, icon_url="https://avatars.githubusercontent.com/u/1633844")
    out.set_timestamp()

    out.add_embed_field(name="Prompt", value=prompt, inline=False)
    out.add_embed_field(name="Session Step", value=training_step)
    out.add_embed_field(name="Global Step", value=global_step)

    attachment_bytes = image.tobytes()
    attachment_id = hashlib.sha1(attachment_bytes).hexdigest()
    attachment_name = f"{attachment_id}.png"

    discord.add_file(file=attachment_bytes, filename=attachment_name)
    out.set_image(f"attachment://{attachment_name}")

    discord.add_embed(out)
    discord.execute()
