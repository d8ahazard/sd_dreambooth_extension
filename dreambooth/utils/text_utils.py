import json
import os
import re
from typing import List

import torch
from transformers import CLIPTextModel

from dreambooth.dataclasses.prompt_data import PromptData


# Implementation from https://github.com/bmaltais/kohya_ss
def encode_hidden_state(text_encoder: CLIPTextModel, input_ids, pad_tokens, b_size, max_token_length,
                        tokenizer_max_length, clip_skip):
    if pad_tokens:
        input_ids = input_ids.reshape((-1, tokenizer_max_length))  # batch_size*3, 77

    if clip_skip <= 1:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out['hidden_states'][-clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    if not pad_tokens:
        return encoder_hidden_states

    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if max_token_length > 75:
        sts_list = [encoder_hidden_states[:, 0].unsqueeze(1)]
        for i in range(1, max_token_length, tokenizer_max_length):
            sts_list.append(encoder_hidden_states[:, i:i + tokenizer_max_length - 2])
        sts_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
        encoder_hidden_states = torch.cat(sts_list, dim=1)

    return encoder_hidden_states


def prompt_to_tags(src_prompt: str, instance_token: str = None, class_token: str = None) -> List[str]:
    src_tags = src_prompt.split(',')
    if class_token:
        conjunctions = ['a ', 'an ', 'the ']
        src_tags = [tag.replace(conjunction + class_token, class_token) for tag in src_tags for conjunction in conjunctions]
    if class_token and instance_token:
        src_tags = [tag.replace(class_token, instance_token) for tag in src_tags]
    src_tags = [' '.join(tag.split()) for tag in src_tags]
    src_tags = [tag.strip() for tag in src_tags if tag]
    if instance_token and instance_token not in src_tags:
        # Prepend instance token to prompt
        src_tags = [instance_token] + src_tags
    return src_tags


def build_strict_tokens(
        caption: str = '',
        tenc_start_token: str = '',
        tenc_end_token: str = ''
):
    caption_list = []
    caption_split = re.split(r'[,;.!?]\s', caption)

    for cap in caption_split:
        words_with_special_token = []
        split_cap = cap.split(" ")

        for sc in split_cap:
            if sc: words_with_special_token.append(f"{sc}</w>")

        new_cap = ' '.join(words_with_special_token)
        caption_list.append(f"{tenc_start_token}{new_cap}{tenc_end_token}")

    special_caption = ', '.join(caption_list)

    return special_caption

def save_token_counts(model_info, instance_prompts: List[PromptData], min_tokens: int):
    token_counts = {}
    for prompt_data in instance_prompts:
        prompt = prompt_data.prompt
        instance_token = prompt_data.instance_token
        class_token = prompt_data.class_token
        prompt_tokens = prompt_to_tags(prompt, instance_token, class_token)
        for token in prompt_tokens:
            token = token.strip()
            if token not in token_counts:
                token_counts[token] = 0
            token_counts[token] += 1
    token_counts = {k: v for k, v in sorted(token_counts.items(), key=lambda item: item[1], reverse=True)}
    counts_to_save = {}
    for token, count in token_counts.items():
        if count < min_tokens:
            continue
        counts_to_save[token] = count

    token_counts_path = os.path.join(model_info.model_dir, "token_counts.json")
    token_data = {model_info.model_name: counts_to_save}
    with open(token_counts_path, "w") as f:
        json.dump(token_data, f, indent=2)
