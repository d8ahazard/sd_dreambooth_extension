from typing import Optional, Union
from io import BytesIO
from PIL import Image
import random
import string
import sys


def is_discord_available():
    try:
        if 'discord_webhook' in sys.modules:
            return True

        import discord_webhook
        return True
    except ImportError:
        return False


def send_update(url: str, image, modal_name: str, prompt: str, seed: Union[str, int], global_step: Union[str, int],  footer: Optional[str] = None):
    from discord_webhook import DiscordWebhook, DiscordEmbed

    webhook = DiscordWebhook(url=url, username="DreamBooth")
    image_embed = DiscordEmbed(color="C70039")

    image_embed.set_author(
        name=modal_name,
        # d8ahazard github avatar for prosperiety
        icon_url='https://avatars.githubusercontent.com/u/1633844'
    )

    image_embed.set_timestamp()
    image_embed.add_embed_field(
        name='Prompt',
        inline=False,
        value=prompt
    )

    image_embed.add_embed_field(
        name='Seed',
        value=seed
    )

    image_embed.add_embed_field(
        name='Step',
        value=global_step
    )

    if footer != None:
        image_embed.set_footer(text=footer)

    def img2bytes(image):
        with BytesIO() as output:
            image.save(output, format="PNG")
            return output.getvalue()

    embed_filename = ''.join(random.choices(
        string.ascii_lowercase + string.ascii_uppercase, k=24
    )) + ".png"

    webhook.add_file(
        file=img2bytes(image),
        filename=embed_filename
    )

    image_embed.set_image(f"attachment://{embed_filename}")
    webhook.add_embed(image_embed)
    webhook.execute()
