function start_training_dreambooth(){
    requestProgress('db');
    gradioApp().querySelector('#db_error').innerHTML='';
    return args_to_array(arguments);
}

onUiUpdate(function(){
    check_progressbar('db', 'db_progressbar', 'db_progress_span', '', 'db_interrupt', 'db_preview', 'db_gallery')
});

ex_titles = titles;

console.log("Existing titles: ", ex_titles);

new_titles = {
    "Model": "The model to train.",
    "Generate Ckpt": "Generate a checkpoint at the current training lvel.",
    "Load Params": "Load model paramters from the last training session.",
    "Cancel": "Cancel training.",
    "Train": "Start training.",
    "Create Model": "Create a new model.",
    "Name": "The name of the model to create.",
    "Import Model from Huggingface Hub": "Import a model from Huggingface.co instead of using a local checkpoint.",
    "Model Path": "The URL to the model on huggingface. Should be in the format of 'developer/model_name'.",
    "HuggingFace Token": "Your huggingface token to use for cloning files.",
    "Source Checkpoint": "The source checkpoint to extract for training.",
    "Scheduler": "Schedule to use for official training stuff.",
    "Create": "Create the danged model already.",
    "Use Concepts List": "Train multiple concepts from a JSON file or string.",
    "Concepts List": "The path to the concepts JSON file, or a JSON string.",
    "Instance Prompt": "A prompt describing the subject. Use [Filewords] to parse image filename or .txt to insert existing prompt here.",
    "Dataset Directory": "The directory containing training images.",
    "Class Prompt": "A prompt for generating classification/regularization images. See the readme for more info.",
    "Classification Dataset Directory": "The directory containing classification/regularization images.",
    "Training Steps": "Total number of training steps to perform. If provided, overrides num_train_epochs.",
    "Total Number of Class/Reg Images": "Total number of classification/regularization images to use. If no images exist, they will be generated. Set to 0 to disable prior preservation.",
    "Learning Rate": "The rate at which the model learns. Default is 0.000005. Use a lower value like 0.000002 or 0.000001 for more complex subjects...like people.",
    "Resolution": "The resolution of input images. You probably want to pre-process them to match this.",
    "Save Checkpoint Frequency": "Save a checkpoint every N steps. ",
    "Save Preview(s) Frequency": "Generate preview images every N steps.",
    "Sample Image Prompt": "The prompt to use when generating preview images.",
    "Sample Image Negative Prompt": "A negative prompt to use when generating preview images.",
    "Number of Samples to Generate": "How many samples to generate per subject.",
    "Sample Seed": "The seed to use when generating samples. Set to -1 to use a random seed every time.",
    "Sample Cfg Scale": "The Classifier-Free Guidance Scale to use for preview images.",
    "Sample Steps": "Number of sampling steps to use when generating preview images.",
    "Auto-Adjust": "Attempt to automatically set training parameters based on total VRAM. Still under development.",
    "Batch Size": "How many images to process at once per training step?",
    "Class Batch Size": "How many classifier/regularization images to generate at once.",
    "Classification Image Negative Prompt": "A negative prompt to use when generating class images. Can be empty.",
    "Classification CFG Scale": "The Classifier-Free Guidance Scale to use for classifier/regularization images.",
    "Classification Steps": "The number of steps to use when generating classifier/regularization images.",
    "Use CPU Only (SLOW)": "Guess what - this will be incredibly slow, but it will work for < 8GB GPUs.",
    "Gradient Checkpointing": "Don't checkpoint the gradients, Duh. Set to False to slightly increase speed at the cost of a bit of VRAM.",
    "Mixed Precision": "You probably want this to be 'fp16'.",
    "Don't Cache Latents": "When this box is *checked* latents will not be cached. When latents are not cached, you will save a bit of VRAM, but train slightly slower",
    "Train Text Encoder": "Enabling this will provide better results and editability, but cost more VRAM.",
    "Train EMA": "Enabling this will provide better results and editability, but cost more VRAM.",
    "Use 8bit Adam": "Enable this to save VRAM.",
    "Gradient Accumulation Steps": "Set this to 2 to increase speed?",
    "Center Crop": "If an image is too large, crop it from the center.",
    "Apply Horizontal Flip": "Randomly decide to flip images horizontally.",
    "Scale Learning Rate": "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    "Learning Rate Scheduler": "The learning rate scheduler to use.",
    "Training Wizard (Person)": "Calculate training parameters for a human subject. Enables prior preservation.",
    "Training Wizard (Object/Style)": "Calculate training parameters for a non-human subject. Disables prior preservation.",
    "# Training Epochs": "Set this or number of steps to train for, not both.",
    "Adam Beta 1": "The beta1 parameter for the Adam optimizer.",
    "Adam Beta 2": "The beta2 parameter for the Adam optimizer.",
    "Adam Weight Decay": "Weight decay for the Adam optimizer, duh.",
    "Adam Epsilon": "",
    "Max Grad Norms": "Max Gradient norms.",
    "Warmup Steps": "Number of steps for the warmup in the lr scheduler.",
    "Pad Tokens": "Pad the input images token lenght to this amount. You probably want to do this.",
    "Max Token Length": "Maximum token length to respect. You probably want to leave this at 75."
}

ex_titles = Object.assign({}, ex_titles, new_titles);
titles = ex_titles;

