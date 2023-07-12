import re
from typing import List

import torch
from transformers import CLIPTextModel

from dreambooth.dataclasses.training_config import TrainingConfig


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


def get_hidden_states(
        args: TrainingConfig, input_ids1, input_ids2, tokenizer1, tokenizer2, text_encoder1, text_encoder2,
        weight_dtype=None
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer
    pool2 = enc_out["text_embeds"]

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if args.max_token_length is None else args.max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if args.max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, args.max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i: i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, args.max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i: i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
            #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
            states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
        hidden_states2 = torch.cat(states_list, dim=1)

        # pool はnの最初のものを使う
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def prompt_to_tags(src_prompt: str, instance_token: str = None, class_token: str = None) -> List[str]:
    src_tags = src_prompt.split(',')
    if class_token:
        conjunctions = ['a ', 'an ', 'the ']
        src_tags = [tag.replace(conjunction + class_token, '') for tag in src_tags for conjunction in conjunctions]
    if class_token and instance_token:
        src_tags = [tag.replace(instance_token, '').replace(class_token, '') for tag in src_tags]
    src_tags = [' '.join(tag.split()) for tag in src_tags]
    src_tags = [tag.strip() for tag in src_tags if tag]
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
