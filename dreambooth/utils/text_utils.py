import re
from typing import List

import torch
import torch.utils.checkpoint
from extensions.sd_dreambooth_extension.dreambooth.dataset.text_data import rare_tokens_list
from transformers import CLIPTextModel, AutoTokenizer


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

def process_tokens(new_tokens_list: str):
    print("Processing your tokens to add.")
    parsed_tokens = new_tokens_list.split(",")
    return [x.strip() for x in parsed_tokens] 

def save_tokens_list(tl_path: str, tl_file: str, new_tokens_list: str):
    os.makedirs(tl_path, exist_ok=True)
    with open(tl_file, "w") as f:
        f.write(new_tokens_list)

def save_text_models(text_encoder, tokenizer, args):
    print("Saving text encoder and tokenizer with new words.")
    text_encoder_path = os.path.join(args.pretrained_model_name_or_path, "text_encoder")
    tokenizer_path = os.path.join(args.pretrained_model_name_or_path, "tokenizer")
    text_encoder.save_pretrained(text_encoder_path)
    tokenizer.save_pretrained(tokenizer_path)

def reload_text_models(text_encoder, tokenizer, args):
    print("Reloading text encoder and tokenizer.")
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            torch_dtype=torch.float32
        )
        
    tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "tokenizer"),
            revision=args.revision,
            use_fast=False,
        )

def add_new_tokens(text_encoder, tokenizer, new_tokens_list: str = "", args = None):
    import random

    original_tokenizer_length = len(tokenizer)
    used_rare_tokens_list = []
    vocab = tokenizer.encoder.keys()
    word_check_counter = 0
    
    if new_tokens_list is not None and bool(new_tokens_list.strip()):
        unique_tokens = process_tokens(new_tokens_list)
        
        for token in unique_tokens:
        
            # Make sure that the new token is one word.
            new_token = f"{token}</w>"
            
            # Check if the token has already been added.
            if new_token in list(vocab):
                print(f"The token: {token} has already beem added. Skipping.")
                continue
                
            # Set bool to check if rare token was used for this token.
            used_rare_token = False
            
            # Find rare word in available list of tokens.
            while used_rare_token == False:
                word_to_replace = random.choice(rare_tokens_list)
                word_check_counter += 1
                
                if word_to_replace not in used_rare_tokens_list and word_to_replace in vocab:
                    used_rare_token = True
                    used_rare_tokens_list.append(word_to_replace)
                
                if word_check_counter >= len(rare_tokens_list):
                    print("All tokens used. Cannot continue adding tokens.")
                    print("Please consider creating a new model, or backup add more rare words to `text_data.py`")
                    break
                    
            # Replace the rare word in the tokenizer.
            print(f"Removing {word_to_replace} and replacing it with {token}.")
            replace_word_val = tokenizer.encoder[word_to_replace]
            tokenizer.encoder[new_token] = tokenizer.encoder.pop(word_to_replace)
            tokenizer.encoder[new_token] = replace_word_val
            
                    
        # Check the length of the tokenizer to ensure we resized it correctly, and save models.
        if original_tokenizer_length == len(tokenizer):
            tl_name, tl_path, tl_file = get_token_list_paths(args)
            save_tokens_list(tl_path, tl_file, new_tokens_list)
            save_text_models(text_encoder, tokenizer, args)
            reload_text_models(text_encoder, tokenizer, args)
            
        # Must be the same size as the original tokenizer, or else you won't be able to use the model as a CKPT.
        else:
            print("Text models not saved. An error occured when adding your tokens.")
            print(f"Original Tokenizer Length: {original_tokenizer_length}")
            print(f"New Tokenizer Length: {len(tokenizer)}")

    else:
        print("No tokens to add.")

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