function db_start_progress(){
    requestProgress('db');
    gradioApp().querySelector('#db_error').innerHTML='';
    gradioApp().querySelector('#db_status').innerHTML='';
    return args_to_array(arguments);
}

function db_save() {
    console.log("Save click?");
    btn = gradioApp().getElementById("db_save_config");
    if(btn!=null) {
        btn.click();
    } else {
        console.log("Can't find btn, trying for btn2");
        let btn2 = document.getElementById("db_save_config");
        if (btn2 != null) {
            btn2.click();
        } else {
            console.log("Can't find button2 either.")
        }
    }
    gradioApp().querySelector('#db_error').innerHTML='';
    gradioApp().querySelector('#db_status').innerHTML='';
    return args_to_array(arguments);
}

function db_save_start_progress(){
    console.log("Save click?");
    btn = gradioApp().getElementById("db_save_config");
    if(btn!=null) {
        btn.click();
    } else {
        console.log("Can't find btn, trying for btn2");
        let btn2 = document.getElementById("db_save_config");
        if (btn2 != null) {
            btn2.click();
        } else {
            console.log("Can't find button2 either.")
        }
    }
    console.log("Saving console.")
    requestProgress('db');
    gradioApp().querySelector('#db_error').innerHTML='';
    gradioApp().querySelector('#db_status').innerHTML='';
    return args_to_array(arguments);
}

onUiUpdate(function(){
    check_progressbar('db', 'db_progressbar', 'db_progress_span', '', 'db_interrupt', 'db_preview', 'db_gallery')
});

ex_titles = titles;

console.log("Existing titles: ", ex_titles);

new_titles = {
    "Adam Beta 1": "The beta1 parameter for the Adam optimizer.",
    "Adam Beta 2": "The beta2 parameter for the Adam optimizer.",
    "Adam Epsilon": "",
    "Adam Weight Decay": "Weight decay for the Adam optimizer, duh.",
    "Apply Horizontal Flip": "Randomly decide to flip images horizontally.",
    "Batch Size": "How many images to process at once per training step?",
    "Cancel": "Cancel training.",
    "Center Crop": "If an image is too large, crop it from the center.",
    "Class Batch Size": "How many classifier/regularization images to generate at once.",
    "Class Prompt": "A prompt for generating classification/regularization images. See the readme for more info.",
    "Class Token": "When using [filewords], this is the class identifier to use/find in existing prompts. Should be a single word.",
    "Classification CFG Scale": "The Classifier-Free Guidance Scale to use for classifier/regularization images.",
    "Classification Dataset Directory": "The directory containing classification/regularization images.",
    "Classification Image Negative Prompt": "A negative prompt to use when generating class images. Can be empty.",
    "Classification Steps": "The number of steps to use when generating classifier/regularization images.",
    "Concepts List": "The path to the concepts JSON file, or a JSON string.",
    "Create Model": "Create a new model.",
    "Create": "Create the danged model already.",
    "Dataset Directory": "The directory containing training images.",
    "Don't Cache Latents": "When this box is *checked* latents will not be cached. When latents are not cached, you will save a bit of VRAM, but train slightly slower",
    "Existing Prompt Contents": "If using [filewords], this tells the string builder how the existing prompts are formatted.",
    "Generate Ckpt": "Generate a checkpoint at the current training level.",
    "Gradient Accumulation Steps": "Set this to 2 to increase speed?",
    "Gradient Checkpointing": "Don't checkpoint the gradients, Duh. Set to False to slightly increase speed at the cost of a bit of VRAM.",
    "Half Model": "Enable this to generate model with fp16 precision. Results in a smaller checkpoint with minimal loss in quality.",
    "HuggingFace Token": "Your huggingface token to use for cloning files.",
    "Import Model from Huggingface Hub": "Import a model from Huggingface.co instead of using a local checkpoint.",
    "Instance Prompt": "A prompt describing the subject. Use [Filewords] to parse image filename or .txt to insert existing prompt here.",
    "Instance Token": "When using [filewords], this is the instance identifier that is unique to your subject. Should be a single word.",
    "Learning Rate Scheduler": "The learning rate scheduler to use.",
    "Learning Rate Warmup Steps": "Number of steps for the warmup in the lr scheduler.",
    "Learning Rate": "The rate at which the model learns. Default is 0.000005. Use a lower value like 0.000002 or 0.000001 for more complex subjects...like people.",
    "Load Params": "Load model paramters from the last training session.",
    "Max Grad Norms": "Max Gradient norms.",
    "Max Token Length": "Maximum token length to respect. You probably want to leave this at 75.",
    "Memory Attention": "The type of memory attention to use. Selecting 'xformers' requires the --xformers command line argument.",
    "Mixed Precision": "You probably want this to be 'fp16'. If using xformers, you definitely want this to be 'fp16'.",
    "Model Path": "The URL to the model on huggingface. Should be in the format of 'developer/model_name'.",
    "Model": "The model to train.",
    "Name": "The name of the model to create.",
    "Number of Samples to Generate": "How many samples to generate per subject.",
    "Pad Tokens": "Pad the input images token lenght to this amount. You probably want to do this.",
    "Performance Wizard (WIP)": "Attempt to automatically set training parameters based on total VRAM. Still under development.",
    "Prior Loss Weight": "Prior loss weight.",
    "Resolution": "The resolution of input images. You probably want to pre-process them to match this.",
    "Sample Cfg Scale": "The Classifier-Free Guidance Scale to use for preview images.",
    "Sample Image Negative Prompt": "A negative prompt to use when generating preview images.",
    "Sample Image Prompt": "The prompt to use when generating preview images.",
    "Sample Seed": "The seed to use when generating samples. Set to -1 to use a random seed every time.",
    "Sample Steps": "Number of sampling steps to use when generating preview images.",
    "Save Checkpoint Frequency": "Save a checkpoint every N steps. ",
    "Save Preview(s) Frequency": "Generate preview images every N steps.",
    "Scale Learning Rate": "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    "Scheduler": "Schedule to use for official training stuff.",
    "Shuffle After Epoch": "When enabled, will shuffle the dataset after the first epoch. Will enable text encoder training and latent caching (More VRAM).",
    "Source Checkpoint": "The source checkpoint to extract for training.",
    "Total Number of Class/Reg Images": "Total number of classification/regularization images to use. If no images exist, they will be generated. Set to 0 to disable prior preservation.",
    "Train EMA": "Enabling this will provide better results and editability, but cost more VRAM.",
    "Train Text Encoder": "Enabling this will provide better results and editability, but cost more VRAM.",
    "Training Epochs": "Set this or number of steps to train for, not both.",
    "Train": "Start training.",
    "Training Steps": "Total number of training steps to perform. If provided, overrides num_train_epochs.",
    "Training Wizard (Object/Style)": "Calculate training parameters for a non-human subject based on number of instance images and set larning rate. Disables prior preservation.",
    "Training Wizard (Person)": "Calculate training parameters for a human subject based on number of instance images and set learning rate. Enables prior preservation.",
    "Use 8bit Adam": "Enable this to save VRAM.",
    "Use CPU Only (SLOW)": "Guess what - this will be incredibly slow, but it will work for < 8GB GPUs.",
    "Use Concepts List": "Train multiple concepts from a JSON file or string."
}

ex_titles = Object.assign({}, ex_titles, new_titles);
titles = ex_titles;

