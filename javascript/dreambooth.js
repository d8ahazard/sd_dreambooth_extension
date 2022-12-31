// Save our current training params before doing a thing

const id_part = "db_";
let params_loaded = false;

// Click a button. Whee.
function save_config() {
    let btn = gradioApp().getElementById("db_save_config");
    if (btn == null) return;
    let do_save = true;
    if (params_loaded === false) {
        do_save = confirm("Warning: You are about to save model parameters that may be empty or from another model. This may erase or overwrite existing settings. If you still wish to continue, click 'OK'.");
    }
    if (do_save === true) {
        console.log("Saving config...");
        btn.click();
    } else {
        console.log("Saving canceled.")
    }
}

function check_save() {
    let do_save = true;
    if (params_loaded === false) {
        do_save = confirm("Warning: You are about to save model parameters that may be empty or from another model. This may erase or overwrite existing settings. If you still wish to continue, click 'OK'.");
    }
    if (do_save === true) {
        console.log("Saving config...");
        let filtered = filterArgs(arguments.length, arguments);
        console.log("Filtered args: ", filtered, filtered.length);
        let status = getRealElement("db_status");
        status.innerHTML = "Config saved.";
        return filtered;
    } else {
        console.log("Saving canceled.")
        return null;
    }
}

function clear_loaded() {
    params_loaded = false;
    return filterArgs(1, arguments);
}

function update_params() {
    if (params_loaded === false) {
        params_loaded = true;
    }
    setTimeout(function(){
        let btn = gradioApp().getElementById("db_update_params");
        if (btn == null) return;
        btn.click();
    },500);
}

function getRealElement(selector) {
    let elem = gradioApp().getElementById(selector);
    if (elem) {
    let child = elem.querySelector('#' + selector);
        if (child) {
            return child;
        } else {
            return elem;
        }
    }
    return elem;
}

// Handler to start save config, progress bar, and filtering args.
function db_start(numArgs, save, startProgress, args) {
    if (save) save_config();
    if (startProgress) requestDbProgress();
    let items = ['db_status', 'db_prompt_list', 'db_gallery_prompt', "db_progressbar"];
    for (let elem in items) {
        let sel = items[elem];
        let outcomeDiv = getRealElement(sel);
        if (outcomeDiv) {
            outcomeDiv.innerHTML = '';
        }
    }


    return filterArgs(numArgs, args);
}

function db_start_sample() {
    return db_start(11, false, true, arguments);
}

// Performance wizard
function db_start_pwizard() {
    return db_start(1, false, false, arguments);
}

// Training wizard
function db_start_twizard() {
    return db_start(1, true, false, arguments);
}

// Generate checkpoint
function db_start_checkpoint() {
    return db_start(7, false, true, arguments);
}

// Generate sample prompts
function db_start_prompts() {
    return db_start(1, true, false, arguments);
}

function db_start_load_params() {
    update_params();
    return db_start(1, false, false, arguments);
}

// Create new checkpoint
function db_start_create() {
    return db_start(7, false, true, arguments);
}

// Train!
function db_start_train() {
    return db_start(8, true, true, arguments);
}

// Generate class images
function db_start_classes() {
    return db_start(5, true, true, arguments);
}

// Return only the number of arguments given as an input
function filterArgs(argsCount, arguments) {
    let args_out = [];
    if (arguments.length >= argsCount && argsCount !== 0) {
        for (let i = 0; i < argsCount; i++) {
            args_out.push(arguments[i]);
        }
    }
    return args_out;
}

let db_titles = {
    "Amount of time to pause between Epochs (s)": "When 'Pause After N Epochs' is greater than 0, this is the amount of time, in seconds, that training will be paused for",
    "Apply Horizontal Flip": "Randomly decide to flip images horizontally.",
    "Batch Size": "How many images to process at once per training step?",
    "Cache Latents": "When this box is checked latents will be cached. When latents are cached, you will save a bit of VRAM, but may train slightly slower.",
    "Cancel": "Cancel training.",
    "Center Crop": "If an image is too large, crop it from the center.",
    "Class Batch Size": "How many classifier/regularization images to generate at once.",
    "Class Prompt": "A prompt for generating classification/regularization images. See the readme for more info.",
    "Class Token": "When using [filewords], this is the class identifier to use/find in existing prompts. Should be a single word.",
    "Classification CFG Scale": "The Classifier-Free Guidance Scale to use for classifier/regularization images.",
    "Classification Dataset Directory": "The directory containing classification/regularization images.",
    "Classification Image Negative Prompt": "A negative prompt to use when generating class images. Can be empty.",
    "Classification Steps": "The number of steps to use when generating classifier/regularization images.",
    "Clip Skip": "Use output of nth layer from back of text encoder (n>=1)",
    "Concepts List": "The path to the concepts JSON file, or a JSON string.",
    "Create Model": "Create a new model.",
    "Create": "Create the danged model already.",
    "Dataset Directory": "The directory containing training images.",
    "Existing Prompt Contents": "If using [filewords], this tells the string builder how the existing prompts are formatted.",
    "Extract EMA Weights": "If EMA weights are saved in a model, these will be extracted instead of the full Unet. Probably not necessary for training or fine-tuning.",
    "Generate Ckpt": "Generate a checkpoint at the current training level.",
    "Generate Class Images": "Create classification images using training settings without training.",
    "Generate Classification Images Using txt2img": "Use the source checkpoint and TXT2IMG to generate class images.",
    "Generate Graph": "Generate graphs from training logs showing learning rate and loss averages over the course of training.",
    "Generate Sample Images": "Generate sample images using the currently saved diffusers model.",
    "Gradient Accumulation Steps": "Number of updates steps to accumulate before performing a backward/update pass.",
    "Gradient Checkpointing": "Don't checkpoint the gradients, Duh. Set to False to slightly increase speed at the cost of a bit of VRAM.",
    "Half Model": "Enable this to generate model with fp16 precision. Results in a smaller checkpoint with minimal loss in quality.",
    "HuggingFace Token": "Your huggingface token to use for cloning files.",
    "Import Model from Huggingface Hub": "Import a model from Huggingface.co instead of using a local checkpoint.",
    "Instance Prompt": "A prompt describing the subject. Use [Filewords] to parse image filename/.txt to insert existing prompt here.",
    "Instance Token": "When using [filewords], this is the instance identifier that is unique to your subject. Should be a single word.",
    "Learning Rate Scheduler": "The learning rate scheduler to use. All schedulers use the provided warmup time except for 'constant'.",
    "Learning Rate Warmup Steps": "Number of steps for the warmup in the lr scheduler.",
    "Learning Rate": "The rate at which the model learns. Default is 2e-6.",
    "Load Params": "Load last saved training parameters for the model..",
    "Log Memory": "Log the current GPU memory usage.",
    "Lora Model": "The Lora model to load for continued fine-tuning or checkpoint generation.",
    "Lora UNET Learning Rate": "The learning rate at which to train lora unet. Regular learning rate is ignored.",
    "Lora Text Learning Rate": "The learning rate at which to train lora text encoder. Regular learning rate is ignored.",
    "Lora Text Weight": "What percentage of the lora weights should be applied to the text encoder when creating a checkpoint.",
    "Lora Weight": "What percentage of the lora weights should be applied to the unet when creating a checkpoint.",
    "Maximum Training Steps": "The max number of steps to train this image for. Set to -1 to train for the general number of max steps",
    "Max Token Length": "Maximum token length to respect. You probably want to leave this at 75.",
    "Max Training Steps": "Total number of training steps to perform. If provided, overrides 'Training Steps Per Image (Epochs)'. Set to 0 to use Steps Per Image.",
    "Memory Attention": "The type of memory attention to use. Selecting 'xformers' requires the --xformers command line argument.",
    "Mixed Precision": "You probably want this to be 'fp16'. If using xformers, you definitely want this to be 'fp16'.",
    "Model Path": "The URL to the model on huggingface. Should be in the format of 'developer/model_name'.",
    "Model": "The model to train.",
    "Name": "The name of the model to create.",
    "Number of Hard Resets": "Number of hard resets of the lr in cosine_with_restarts scheduler.",
    "Number of Samples to Generate": "How many samples to generate per subject.",
    "Pad Tokens": "Pad the input images token length to this amount. You probably want to do this.",
    "Pause After N Epochs": "Number of epochs after which training will be paused for the specified time. Useful if you want to give your GPU a rest.",
    "Performance Wizard (WIP)": "Attempt to automatically set training parameters based on total VRAM. Still under development.",
    "Polynomial Power": "Power factor of the polynomial scheduler.",
    "Preview Prompts": "Generate a JSON representation of prompt data used for training.",
    "Prior Loss Weight": "Prior loss weight.",
    "Resolution": "The resolution of input images. You probably want to pre-process them to match this.",
    "Sample CFG Scale": "The Classifier-Free Guidance Scale to use for preview images.",
    "Sample Image Prompt": "The prompt to use when generating preview images.",
    "Sample Prompt": "The prompt to use to generate a sample image",
    "Sample Prompt Template File": "The path to a txt file to use for sample prompts. Use [filewords] or [name] to insert class token in sample prompts",
    "Sample Negative Prompt": "A negative prompt to use when generating preview images.",
    "Sample Seed": "The seed to use when generating samples. Set to -1 to use a random seed every time.",
    "Sample Steps": "The number of steps to use when generating classifier/regularization images.",
    "Sanity Sample Prompt": "A prompt used to generate a 'baseline' image that will be created with other samples to verify model fidelity.",
    "Sanity Sample Seed": "The seed to use when generating the validation sample image. -1 is not supported.",
    "Save Checkpoint to Subdirectory": "When enabled, checkpoints will be saved to a subdirectory in the selected checkpoints folder.",
    "Save Params": "Save the current training parameters to the model config file.",
    "Save Model Frequency (Epoch)": "Save a checkpoint every N epochs.",
    "Save Preview(s) Frequency (Epoch)": "Generate preview images every N epochs.",
    "Save Model Frequency (Step)": "Save a checkpoint every N epochs. Must be divisible by batch number.",
    "Save Preview(s) Frequency (Step)": "Generate preview images every N steps. Must be divisible by batch number.",
    "Save Weights": "Save weights/checkpoint/snapshot as specified in the saving section for saving 'during' training.",
    "Scheduler": "Schedule to use for official training stuff.",
    "Set Gradients to None When Zeroing": "Can increase training speed at the cost of a slight increase in VRAM usage.",
    "Shuffle After Epoch": "When enabled, will shuffle the dataset after the first epoch. Will enable text encoder training and latent caching (More VRAM).",
    "Source Checkpoint": "The source checkpoint to extract for training.",
    "Text Encoder Epochs": "The number of steps per image (Epoch) to train the text encoder for. Set to -1 to train the same amount as the Unet.",
    "Total Number of Class/Reg Images": "Total number of classification/regularization images to use. If no images exist, they will be generated. Set to 0 to disable prior preservation.",
    "Train Text Encoder": "Enabling this will provide better results and editability, but cost more VRAM.",
    "Train": "Start training.",
    "Train Imagic Only": "Uses Imagic for training instead of full dreambooth, useful for training with a single instance image.",
    "Training Steps Per Image (Epochs)": "Set this or number of steps to train for, not both. Epochs = 'The number of steps to run per image.'",
    "Training Wizard (Object/Style)": "Calculate training parameters for a non-human subject based on number of instance images and set larning rate. Disables prior preservation.",
    "Training Wizard (Person)": "Calculate training parameters for a human subject based on number of instance images and set learning rate. Enables prior preservation.",
    "Use 8bit Adam": "Enable this to save VRAM.",
    "Use CPU Only (SLOW)": "Guess what - this will be incredibly slow, but it will work for < 8GB GPUs.",
    "Use Concepts List": "Train multiple concepts from a JSON file or string.",
    "Use Epoch Values for Save Frequency": "When enabled, save frequencies below are based on number of epochs. When disabled, frequencies are based on number of training steps.",
    "Use Lifetime Epochs When Saving": "When checked, will save preview images and checkpoints using lifetime epochs, versus current training epochs.",
    "Use Lifetime Steps When Saving": "When checked, will save preview images and checkpoints using lifetime steps, versus current training steps.",
    "Use EMA": "Enabling this will provide better results and editability, but cost more VRAM.",
    "Use LORA": "Uses Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning. Uses less VRAM, saves a .pt file instead of a full checkpoint"
}

// Do a thing when the UI updates
onUiUpdate(function () {
    db_progressbar();

    gradioApp().querySelectorAll('span, button, select, p').forEach(function (span) {
        let tooltip = db_titles[span.textContent];
        if (!tooltip) {
            tooltip = db_titles[span.value];
        }

        if (!tooltip) {
            for (const c of span.classList) {
                if (c in db_titles) {
                    tooltip = db_titles[c];
                    break;
                }
            }
        }

        if (tooltip) {
            span.title = tooltip;
        }

    });

    gradioApp().querySelectorAll('select').forEach(function (select) {
        if (select.onchange != null) return;
        select.onchange = function () {
            select.title = db_titles[select.value] || "";
        }
    });

    gradioApp().querySelectorAll('.gallery-item').forEach(function (btn) {
        if (btn.onchange != null) return;
        btn.onchange = function() {
            // Dummy function so we don't keep setting up the observer.
        }
        checkPrompts();
        const options = {
            attributes: true
        }

        function callback(mutationList, observer) {
            mutationList.forEach(function (mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    checkPrompts();
                }
            });
        }

        const observer = new MutationObserver(callback);
        observer.observe(btn, options);

    });
});

function checkPrompts() {
    let prevSelectedIndex = selected_gallery_index();
    let desc_list = getRealElement("db_prompt_list");
    let des_box = getRealElement("db_gallery_prompt");
    let prompts = desc_list.innerHTML;
    if (prompts.includes("<br>")) {
        let prompt_list = prompts.split("<br>");
        if (prevSelectedIndex !== -1 && prevSelectedIndex < prompt_list.length) {
            des_box.innerHTML = prompt_list[prevSelectedIndex];
        }
    }
}

let progressTimeout = null;
let galleryObserver = null;
let gallerySet = false;

function db_progressbar(){
    let id_gallery = "db_gallery";

    // gradio 3.8's enlightened approach allows them to create two nested div elements inside each other with same id
    // every time you use gr.HTML(elem_id='xxx'), so we handle this here
    let progressbar = gradioApp().querySelector("#db_progressbar #db_progressbar");
    let progressbarParent;
    if(progressbar){
        progressbarParent = gradioApp().querySelector("#db_progressbar");
    } else{
        progressbar = gradioApp().getElementById("db_progressbar");
        progressbarParent = null;
    }

    // let skip = id_skip ? gradioApp().getElementById(id_skip) : null;
    let interrupt = gradioApp().getElementById("db_cancel");
    let gen_sample = gradioApp().getElementById("db_train_sample");
    let gen_ckpt = gradioApp().getElementById("db_gen_ckpt");
    let gen_ckpt_during = gradioApp().getElementById("db_gen_ckpt_during")
    let train = gradioApp().getElementById("db_train");
    
    if(progressbar && progressbar.offsetParent){
        if(progressbar.innerText){
            let newtitle = '[' + progressbar.innerText.trim() + '] Stable Diffusion';
            if(document.title !== newtitle){
                document.title =  newtitle;          
            }
        }else{
            let newtitle = 'Stable Diffusion'
            if(document.title !== newtitle){
                document.title =  newtitle;          
            }
        }
    }
    
	if(progressbar != null){
	    let mutationObserver = new MutationObserver(function(m){
            if(progressTimeout) {
                return;
            }

            let preview = gradioApp().getElementById("db_preview");
            let gallery = gradioApp().getElementById("db_gallery");

            if(preview != null && gallery != null){
                preview.style.width = gallery.clientWidth + "px"
                preview.style.height = gallery.clientHeight + "px"
                if(progressbarParent) progressbar.style.width = progressbarParent.clientWidth + "px"

				//only watch gallery if there is a generation process going on
                checkDbGallery();

                let progressDiv = gradioApp().querySelectorAll('#db_progress_span').length > 0;
                if(progressDiv){
                    progressTimeout = window.setTimeout(function() {
                        requestMoreDbProgress();
                    }, 500);
                } else{
                    interrupt.style.display = "none";
                    gen_sample.style.display = "none";
                    gen_ckpt_during.style.display = "none";
                    gen_ckpt.style.display = "block";
                    train.style.display = "block";
			
                    //disconnect observer once generation finished, so user can close selected image if they want
                    if (galleryObserver) {
                        galleryObserver.disconnect();
                        galleryObserver = null;
                        gallerySet = false;
                    }
                }
            }

        });
        mutationObserver.observe( progressbar, { childList:true, subtree:true });
	}
}

function checkDbGallery(){
    if (gallerySet) return;
    let gallery = gradioApp().getElementById("db_gallery");
    // if gallery has no change, no need to setting up observer again.
    if (gallery){
        if(galleryObserver){
            galleryObserver.disconnect();
        }
        // Get the last selected item in the gallery.
        let prevSelectedIndex = selected_gallery_index();

        // Make things clickable?
        galleryObserver = new MutationObserver(function (){
            let galleryButtons = gradioApp().querySelectorAll('#db_gallery .gallery-item');
            let galleryBtnSelected = gradioApp().querySelector('#db_gallery .gallery-item.\\!ring-2');
            if (prevSelectedIndex !== -1 && galleryButtons.length>prevSelectedIndex && !galleryBtnSelected) {
                // automatically re-open previously selected index (if exists)
                let activeElement = gradioApp().activeElement;
                let scrollX = window.scrollX;
                let scrollY = window.scrollY;

                galleryButtons[prevSelectedIndex].click();

                // When the gallery button is clicked, it gains focus and scrolls itself into view
                // We need to scroll back to the previous position
                setTimeout(function (){
                    window.scrollTo(scrollX, scrollY);
                }, 50);

                if(activeElement){
                    // i fought this for about an hour; i don't know why the focus is lost or why this helps recover it
                    // if someone has a better solution please by all means
                    setTimeout(function (){
                        activeElement.focus({
                            preventScroll: true // Refocus the element that was focused before the gallery was opened without scrolling to it
                        })
                    }, 1);
                }
            }
        })
        galleryObserver.observe( gallery, { childList:true, subtree:false });
        gallerySet = true;

    }
}

function requestDbProgress(){
    let btn = gradioApp().getElementById("db_check_progress_initial");
    if(btn==null) {
        console.log("Can't find da button!.")
        return;
    }
    btn.click();
    db_progressbar();
}

function requestMoreDbProgress(){
    let btn = gradioApp().getElementById("db_check_progress");
    if(btn==null) {
        console.log("Check progress button is null!");
        return;
    }
    btn.click();
    progressTimeout = null;
    let progressDiv = gradioApp().querySelectorAll('#db_progress_span').length > 0;
    // TODO: Eventually implement other skip/cancel buttons.
    // let skip = id_skip ? gradioApp().getElementById("db_skip") : null;
    let interrupt = gradioApp().getElementById("db_cancel");
    let train = gradioApp().getElementById("db_train");
    let gen_sample = gradioApp().getElementById("db_train_sample");
    let gen_ckpt = gradioApp().getElementById("db_gen_ckpt");
    let gen_ckpt_during = gradioApp().getElementById("db_gen_ckpt_during");
    if(progressDiv && interrupt && train && gen_sample){
        gen_sample.style.display = "block";
        train.style.display = "none";
        interrupt.style.display = "block";
        gen_ckpt.style.display = "none";
        gen_ckpt_during.style.display = "block";
    }
}

