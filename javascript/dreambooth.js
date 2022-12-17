// Save our current training params before doing a thing

const id_part = "db_";

// Click a button. Whee.
function save_config() {
    let btn = gradioApp().getElementById("db_save_config");
    if (btn == null) return;
    console.log("Saving config...");
    btn.click();
}

function log_save() {
    console.log("Saving: ", arguments);
    return arguments;
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
    console.log("Clearing statuses.");
    let items = ['db_status', 'db_status2'];
    for (let elem in items) {
        let sel = items[elem];
        let outcomeDiv = getRealElement(sel);
        if (outcomeDiv) {
            outcomeDiv.innerHTML = '';
        } else {
            console.log("YOU SUCK: ", sel);
        }

    }


    return filterArgs(numArgs, args);
}

function db_start_sample() {
    return db_start(5, true, true, arguments);
}

function db_start_pwizard() {
    return db_start(0, true, false, arguments);
}

function db_start_twizard() {
    return db_start(1, true, false, arguments);
}

function db_start_checkpoint() {
    return db_start(7, false, true, arguments);
}

function db_start_prompts() {
    return db_start(1, true, false, arguments);
}

function db_start_create() {
    return db_start(7, false, true, arguments);
}

function db_start_train() {
    return db_start(8, true, true, arguments);
}

function db_start_classes() {
    return db_start(4, true, true, arguments);
}

// Return only the number of arguments given as an input
function filterArgs(argsCount, arguments) {
    let args_out = [];
    if (arguments.length > argsCount && argsCount !== 0) {
        for (let i = 0; i < argsCount; i++) {
            args_out.push(arguments[i]);
        }
    }
    console.log("Filtered args ("+argsCount+"): ", args_out);
    return args_out;
}


// Do a thing when the UI updates
onUiUpdate(function () {
    db_progressbar();
});


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
                showGalleryImage();

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
    console.log("Requesting progress start...");
    btn.click();
    db_progressbar();
}

function requestMoreDbProgress(){
    let btn = gradioApp().getElementById("db_check_progress");
    if(btn==null) {
        console.log("Check progress button is null!");
        return;
    }
    console.log("MORE PROGRESS!");
    btn.click();
    progressTimeout = null;
    let progressDiv = gradioApp().querySelectorAll('#db_progress_span').length > 0;
    // TODO: Eventually implement other skip/cancel buttons.
    // let skip = id_skip ? gradioApp().getElementById("db_skip") : null;
    let interrupt = gradioApp().getElementById("db_cancel");
    if(progressDiv && interrupt){
        // if (skip) {
        //     skip.style.display = "block";
        // }
        interrupt.style.display = "block";
    }
}

let ex_titles;
let broke_titles = false;
try {
    ex_titles = titles;
} catch (e) {
    broke_titles = true;
}
if (broke_titles) {
    let titles = {};
}

console.log("Existing titles: ", ex_titles);
let new_titles = {
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
    "Extract EMA Weights": "If EMA weights are saved in a model, these will be extracted instead of the full Unet. Probably not necessary for training or fine-tuning.",
    "Generate Ckpt": "Generate a checkpoint at the current training level.",
    "Generate Sample Images": "Generate sample images using the currently saved diffusers model.",
    "Gradient Accumulation Steps": "Number of updates steps to accumulate before performing a backward/update pass.",
    "Gradient Checkpointing": "Don't checkpoint the gradients, Duh. Set to False to slightly increase speed at the cost of a bit of VRAM.",
    "Half Model": "Enable this to generate model with fp16 precision. Results in a smaller checkpoint with minimal loss in quality.",
    "HuggingFace Token": "Your huggingface token to use for cloning files.",
    "Import Model from Huggingface Hub": "Import a model from Huggingface.co instead of using a local checkpoint.",
    "Instance Prompt": "A prompt describing the subject. Use [Filewords] to parse image filename/.txt to insert existing prompt here.",
    "Instance Token": "When using [filewords], this is the instance identifier that is unique to your subject. Should be a single word.",
    "Learning Rate Scheduler": "The learning rate scheduler to use.",
    "Learning Rate Warmup Steps": "Number of steps for the warmup in the lr scheduler.",
    "Learning Rate": "The rate at which the model learns. Default is 0.000005. Use a lower value like 0.000002 or 0.000001 for more complex subjects...like people.",
    "Load Params": "Load last saved training parameters for the model..",
    "Log Memory": "Log the current GPU memory usage.",
    "Lora Model": "The Lora model to load for continued fine-tuning or checkpoint generation.",
    "Lora Weight": "What percentage of the lora weights should be applied to the unet when training or creating a checkpoint",
    "Maximum Training Steps": "The max number of steps to train this image for. Set to -1 to train for the general number of max steps",
    "Max Token Length": "Maximum token length to respect. You probably want to leave this at 75.",
    "Max Training Steps": "Total number of training steps to perform. If provided, overrides 'Training Steps Per Image (Epochs)'. Set to 0 to use Steps Per Image.",
    "Memory Attention": "The type of memory attention to use. Selecting 'xformers' requires the --xformers command line argument.",
    "Mixed Precision": "You probably want this to be 'fp16'. If using xformers, you definitely want this to be 'fp16'.",
    "Model Path": "The URL to the model on huggingface. Should be in the format of 'developer/model_name'.",
    "Model": "The model to train.",
    "Name": "The name of the model to create.",
    "Number of Samples to Generate": "How many samples to generate per subject.",
    "Pad Tokens": "Pad the input images token lenght to this amount. You probably want to do this.",
    "Performance Wizard (WIP)": "Attempt to automatically set training parameters based on total VRAM. Still under development.",
    "Preview Prompts": "Generate a JSON representation of prompt data used for training.",
    "Prior Loss Weight": "Prior loss weight.",
    "Resolution": "The resolution of input images. You probably want to pre-process them to match this.",
    "Sample CFG Scale": "The Classifier-Free Guidance Scale to use for preview images.",
    "Sample Image Negative Prompt": "A negative prompt to use when generating preview images.",
    "Sample Image Prompt": "The prompt to use when generating preview images.",
    "Sample Prompt": "The prompt to use to generate a sample image",
    "Sample Prompt Template File": "The path to a txt file to use for sample prompts. Use [filewords] or [name] to insert class token in sample prompts",
    "Sample Seed": "The seed to use when generating samples. Set to -1 to use a random seed every time.",
    "Sample Steps": "The number of steps to use when generating classifier/regularization images.",
    "Save Checkpoint to Subdirectory": "When enabled, checkpoints will be saved to a subdirectory in the selected checkpoints folder.",
    "Save Params": "Save the current training parameters to the model config file.",
    "Save Preview/Ckpt Every Epoch": "When enabled, save frequencies below are based on number of epochs. When disabled, frequencies are based on number of training steps.",
    "Save Checkpoint Frequency": "Save a checkpoint every N steps. ",
    "Save Preview(s) Frequency": "Generate preview images every N steps.",
    "Scale Learning Rate": "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    "Scheduler": "Schedule to use for official training stuff.",
    "Shuffle After Epoch": "When enabled, will shuffle the dataset after the first epoch. Will enable text encoder training and latent caching (More VRAM).",
    "Source Checkpoint": "The source checkpoint to extract for training.",
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
    "Use Lifetime Steps/Epochs When Saving": "When checked, will save preview images and checkpoints using lifetime steps/epochs, versus current training steps.",
    "Use EMA": "Enabling this will provide better results and editability, but cost more VRAM.",
    "Use LORA": "Uses Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning. Uses less VRAM, saves a .pt file instead of a full checkpoint"
}
ex_titles = Object.assign({}, ex_titles, new_titles);
titles = ex_titles;

