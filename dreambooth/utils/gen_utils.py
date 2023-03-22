import os
import traceback
from typing import List

from accelerate import Accelerator
from transformers import AutoTokenizer

from dreambooth import shared
from dreambooth.dataclasses.db_config import DreamboothConfig, from_file
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataset.class_dataset import ClassDataset
from dreambooth.dataset.db_dataset import DbDataset
from dreambooth.shared import status
from dreambooth.utils.image_utils import db_save_image
from dreambooth.utils.utils import cleanup
from helpers.image_builder import ImageBuilder
from helpers.mytqdm import mytqdm


def generate_dataset(model_name: str, instance_prompts: List[PromptData] = None, class_prompts: List[PromptData] = None,
                     batch_size=None, tokenizer=None, vae=None, debug=True, model_dir=""):
    if debug:
        print("Generating dataset.")
    from dreambooth.ui_functions import gr_update

    db_gallery = gr_update(value=None)
    db_prompt_list = gr_update(value=None)
    db_status = gr_update(value=None)

    args = from_file(model_name)

    if batch_size is None:
        batch_size = args.train_batch_size

    if args is None:
        print("No CONFIG!")
        return db_gallery, db_prompt_list, db_status

    if debug and tokenizer is None:
        print("Definitely made a tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "tokenizer"),
            revision=args.revision,
            use_fast=False,
        )

    tokens = []

    print(f"Found {len(class_prompts)} reg images.")

    print("Preparing dataset...")

    if args.strict_tokens: print("Building prompts with strict tokens enabled.")

    train_dataset = DbDataset(
        batch_size=batch_size,
        instance_prompts=instance_prompts,
        class_prompts=class_prompts,
        tokens=tokens,
        tokenizer=tokenizer,
        resolution=args.resolution,
        hflip=args.hflip,
        shuffle_tags=args.shuffle_tags,
        strict_tokens=args.strict_tokens,
        not_pad_tokens=not args.pad_tokens,
        debug_dataset=debug,
        model_dir=model_dir
    )
    train_dataset.make_buckets_with_caching(vae)

    # train_dataset = train_dataset.pin_memory()
    print(f"Total dataset length (steps): {len(train_dataset)}")
    return train_dataset


def generate_classifiers(
        args: DreamboothConfig,
        class_gen_method: str = "Native Diffusers",
        accelerator: Accelerator = None,
        ui=True):
    """

    @param args: A DreamboothConfig
    @param class_gen_method
    @param accelerator: An optional existing accelerator to use.
    @param ui: Whether this was called by th UI, or is being run during training.
    @return:
    generated: Number of images generated
    images: A list of images or image paths, depending on if returning to the UI or not.
    if ui is False, this will return a second array of paths representing the class paths.
    """
    out_images = []
    instance_prompts = []
    class_prompts = []
    try:
        status.textinfo = "Preparing dataset..."
        prompt_dataset = ClassDataset(args.concepts(), args.model_dir, args.resolution, False)
        instance_prompts = prompt_dataset.instance_prompts
        class_prompts = prompt_dataset.class_prompts
    except Exception as p:
        print(f"Exception generating dataset: {str(p)}")
        traceback.print_exc()
        if ui:
            return 0, []
        else:
            return 0, instance_prompts, class_prompts

    set_len = prompt_dataset.__len__()
    if set_len == 0:
        print("Nothing to generate.")
        if ui:
            return 0, []
        else:
            return 0, instance_prompts, class_prompts

    print(f"Generating {set_len} class images for training...")
    pbar = mytqdm(total=set_len, desc=f"Generating class images 0/{set_len}:", position=0)
    shared.status.job_count = set_len
    shared.status.job_no = 0
    builder = ImageBuilder(
        args,
        class_gen_method=class_gen_method,
        lora_model=args.lora_model_name,
        batch_size=args.sample_batch_size,
        accelerator=accelerator,
        lora_unet_rank=args.lora_unet_rank,
        lora_txt_rank=args.lora_txt_rank,
        source_checkpoint=args.src
    )

    generated = 0
    actual_idx = 0
    for i in range(set_len):
        first_res = None
        if status.interrupted or generated >= set_len:
            break
        prompts = []
        # Decrease batch size
        if set_len - generated < args.sample_batch_size:
            batch_size = set_len - generated
        else:
            batch_size = args.sample_batch_size
        for b in range(batch_size):
            # Get the new prompt data
            pd = prompt_dataset.__getitem__(actual_idx)
            if pd is None:
                break
            # Ensure that our image batches have the right resolutions
            if first_res is None:
                first_res = pd.resolution
            if pd.resolution == first_res:
                prompts.append(pd)
                actual_idx += 1
            else:
                break

        new_images = builder.generate_images(prompts, pbar)
        i_idx = 0
        preview_images = []
        preview_prompts = []
        for image in new_images:
            if generated >= set_len:
                break
            try:
                # Retrieve prompt data object
                pd = prompts[i_idx]
                # Save image and get new filename
                image_filename = db_save_image(image, pd)
                # Set filename here for later retrieval
                pd.src_image = image_filename
                # NOW STORE IT
                class_prompts.append(pd)
                if ui:
                    out_images.append(image)
                i_idx += 1
                generated += 1
                pbar.reset(set_len)
                pbar.update(generated)
                pbar.set_description(f"Generating class images {generated}/{set_len}:", True)
                shared.status.job_count = set_len
                preview_images.append(image_filename)
                preview_prompts.append(pd.prompt)
            except Exception as e:
                print(f"Exception generating images: {e}")
                traceback.print_exc()

        status.current_image = preview_images
        status.sample_prompts = preview_prompts
    builder.unload(ui)
    del prompt_dataset
    cleanup()
    print(f"Generated {generated} new class images.")
    if ui:
        return generated, out_images
    else:
        return generated, instance_prompts, class_prompts
