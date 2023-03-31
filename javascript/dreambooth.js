// Save our current training params before doing a thing
let params_loaded = false;
let training_started = false;
let closeBtn;
let modalShown = false;
let locked = false;
let listenersSet = false;
let timeouts = [];
let listeners = {};

function save_config() {
    let btn = gradioApp().getElementById("db_save_config");
    if (btn == null) return;
    let do_save = true;
    if (params_loaded === false) {
        do_save = confirm("Warning: Current UI Params have not been saved. Press 'OK' to save them now, or 'Cancel' to continue without saving.");
    }
    if (do_save === true) {
        btn.click();
        params_loaded = true;
    }
}

function toggleComponents(enable, disableAll) {
    const elements = ['DbTopRow', 'SettingsPanel'];
    if (disableAll) {
        console.log("Disabling all DB elements!");
        elements.push("ModelPanel")
        locked = true;
    }
    elements.forEach(element => {
        let elem = getRealElement(element);
        if (elem === null || elem === undefined) {
            console.log("Can't find element: ", element);
        } else {
            const labels = elem.getElementsByTagName('label');
            const inputs = elem.querySelectorAll("input, textarea, button");

            Array.from(labels).forEach(label => {
                if (enable) {
                    label.classList.remove('!cursor-not-allowed');
                } else {
                    label.classList.add('!cursor-not-allowed');
                }
            });

            Array.from(inputs).forEach(input => {
                if (input.id.indexOf("secret") === -1) {
                    input.disabled = !enable;
                }
            });
        }
    });
}

// Disconnect a gradio mutation observer, update the element value, and reconnect the observer?
function updateInputValue(elements, newValue) {
    const savedListeners = [];
    const savedObservers = [];

    elements.forEach((element) => {
        // Save any existing listeners and remove them
        const listeners = [];
        const events = ['change', 'input'];
        events.forEach((event) => {
            if (element['on' + event]) {
                listeners.push({
                    event,
                    listener: element['on' + event],
                });
                element['on' + event] = null;
            }
            const eventListeners = element.getEventListeners?.(event);
            if (eventListeners) {
                eventListeners.forEach(({ listener }) => {
                    listeners.push({
                        event,
                        listener,
                    });
                    element.removeEventListener(event, listener);
                });
            }
        });
        savedListeners.push(listeners);

        // Save any existing MutationObservers and disconnect them
        const observer = new MutationObserver(() => {
        });
        if (observer && element.tagName === 'INPUT') {
            observer.observe(element, {
                attributes: true,
                attributeFilter: ['value'],
            });
            savedObservers.push(observer);
            observer.disconnect();
        } else {
            savedObservers.push(null);
        }

        // Update the value of the element
        element.value = newValue;
    });

    // Restore any saved listeners and MutationObservers
    savedListeners.forEach((listeners, i) => {
        const element = elements[i];
        listeners.forEach(({ event, listener }) => {
            if (listener) {
                element.addEventListener(event, listener);
            }
        });
    });

    savedObservers.forEach((observer, i) => {
        const element = elements[i];
        if (observer) {
            observer.observe(element, {
                attributes: true,
                attributeFilter: ['value'],
            });
        }
    });
}


// Fix steps on sliders. God this is a lot of work for one stupid thing...
function handleNumberInputs() {
    const numberInputs = gradioApp()
       .querySelector('#tab_dreambooth_interface')
       ?.querySelectorAll('input[type="number"]');
    numberInputs?.forEach((numberInput) => {
        const step = Number(numberInput.step) || 1;
        const parentDiv = numberInput.parentElement;
        const labelFor = parentDiv.querySelector('label');
        if (labelFor) {
            const tgt = labelFor.getAttribute("for");
            if (listeners[tgt]) return;
            const rangeInput = getRealElement(tgt);
            if (rangeInput && rangeInput.type === 'range') {
                let timeouts = [];
                listeners[tgt] = true;
                numberInput.oninput = () => {
                    if (timeouts[tgt]) {
                        clearTimeout(timeouts[tgt]);
                    }
                    timeouts[tgt] = setTimeout(() => {
                        let value = Number(numberInput.value) || 0;
                        const min = parseFloat(rangeInput.min) || 0;
                        const max = parseFloat(rangeInput.max) || 100;
                        if (value < min) {
                            value = min;
                        } else if (value > max) {
                            value = max;
                        }
                        const remainder = value % step;
                        if (remainder !== 0) {
                            value -= remainder;
                            if (remainder >= step / 2) {
                                value += step;
                            }
                        }
                        if (value !== numberInput.value) {
                            numberInput.value = value;
                        }
                    }, 500);
                };

            }
        }
    });
}


function check_save() {
    let do_save = true;
    if (params_loaded === false) {
        do_save = confirm("Warning: You are about to save model parameters that may be empty or from another model. This may erase or overwrite existing settings. If you still wish to continue, click 'OK'.");
    }
    if (do_save === true) {
        let filtered = filterArgs(arguments.length, arguments);
        let status = getRealElement("db_status");
        status.innerHTML = "Config saved."
        params_loaded = true;
        return filtered;
    } else {
        console.log("Saving canceled.")
        return null;
    }
}

function clear_loaded() {
    if (arguments[0] !== "") {
        toggleComponents(true, false);
        let hintRow = getRealElement("hint_row");
        hintRow.style.display = "none";
    }

    params_loaded = false;
    return filterArgs(1, arguments);
}

function update_params() {
    if (params_loaded === false) {
        params_loaded = true;
    }
    setTimeout(function () {
        let btn = gradioApp().getElementById("db_update_params");
        if (btn == null) return;
        btn.click();
    }, 500);
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
    return db_start(18, false, true, arguments);
}

function db_start_crop() {
    return db_start(5, false, true, arguments);
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
    return db_start(1, true, true, arguments);
}

// Generate sample prompts
function db_start_prompts() {
    return db_start(1, true, false, arguments);
}

function db_start_logs() {
    return db_start(2, false, true, arguments);
}

// Debug bucketing
function db_start_buckets() {
    return db_start(3, true, true, arguments);
}

function db_start_load_params() {
    update_params();
    return db_start(1, false, false, arguments);
}

// Create new checkpoint
function db_start_create() {
    clear_loaded();
    return db_start(8, false, true, arguments);
}

// Train!
function db_start_train() {
    training_started = true;
    return db_start(2, true, true, arguments);
}

// Generate class images
function db_start_classes() {
    return db_start(2, true, true, arguments);
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
    "API Key": "Used for securing the Web API. Click the refresh button to the right to (re)generate your key, the trash icon to remove it.",
    "AdamW Weight Decay": "The weight decay of the AdamW Optimizer. Values closer to 0 closely match your training dataset, and values closer to 1 generalize more and deviate from your training dataset. Default is 1e-2, values lower than 0.1 are recommended.",
    "Amount of time to pause between Epochs (s)": "When 'Pause After N Epochs' is greater than 0, this is the amount of time, in seconds, that training will be paused for",
    "Apply Horizontal Flip": "Randomly decide to flip images horizontally.",
    "Batch Size": "How many images to process at once per training step?",
    "Betas": "The betas of the used by the Dadaptation schedulers. Default is 0.9, 0.999.",
    "Cache Latents": "When this box is checked latents will be cached. Caching latents will use more VRAM, but improve training speed.",
    "Cancel": "Cancel training.",
    "Class Batch Size": "How many classifier/regularization images to generate at once.",
    "Class Images Per Instance Image": "How many classification images to use per instance image.",
    "Class Prompt": "A prompt for generating classification/regularization images. See the readme for more info.",
    "Class Token": "When using [filewords], this is the class identifier to use/find in existing prompts. Should be a single word.",
    "Classification CFG Scale": "The Classifier-Free Guidance Scale to use for classifier/regularization images.",
    "Classification Dataset Directory": "The directory containing classification/regularization images.",
    "Classification Image Negative Prompt": "A negative prompt to use when generating class images. Can be empty.",
    "Classification Steps": "The number of steps to use when generating classifier/regularization images.",
    "Clip Skip": "Use output of nth layer from back of text encoder (n>=1)",
    "Concepts List": "The path to the concepts JSON file, or a JSON string.",
    "Constant/Linear Starting Factor": "Sets the initial learning rate to the main_lr * this value. If you had a target LR of .000006 and set this to .5, the scheduler would start at .000003 and increase until it reached .000006.",
    "Create From Hub": "Import a model from Huggingface.co instead of using a local checkpoint. Hub model MUST contain diffusion weights. You can specify a local folder with a cloned model, no HF token will be needed in this case.",
    "Create Model": "Create a new model.",
    "Create": "Create the danged model already.",
    "Custom Model Name": "A custom name to use when saving .ckpt and .pt files. Subdirectories will also be named this.",
    "Dataset Directory": "The directory containing training images.",
    "Debug Buckets": "Examine the instance and class images and report any instance images without corresponding class images.",
    "Decouple": "Decouple the weight decay from learning rate.",
    "Discord Webhook": "Send training samples to a Discord channel after generation.",
    "D0": "Initial D estimate for D-adaptation",
    "Existing Prompt Contents": "If using [filewords], this tells the string builder how the existing prompts are formatted.",
    "EPS": "The epsilon value to use for the Dadaptation optimizers.",
    "Extract EMA Weights": "If EMA weights are saved in a model, these will be extracted instead of the full Unet. Probably not necessary for training or fine-tuning.",
    "Freeze CLIP Normalization Layers": "Keep the normalization layers of CLIP frozen during training. Advanced usage, may increase model performance and editability.",
    "Generate Ckpt": "Generate a checkpoint at the current training level.",
    "Generate Class Images": "Create classification images using training settings without training.",
    "Generate Classification Images Using txt2img": "Use the source checkpoint and TXT2IMG to generate class images.",
    "Generate Classification Images to match Instance Resolutions": "Instead of generating square class images, they will be generated at the same resolution(s) as class images.",
    "Generate Graph": "Generate graphs from training logs showing learning rate and loss averages over the course of training.",
    "Generate Sample Images": "Generate sample images using the currently saved diffusers model.",
    "Generate Samples": "Trigger sample generation after the next training epoch.",
    "Generate a .ckpt file when saving during training.": "When enabled, a checkpoint will be generated at the specified epoch intervals while training is active. This also controls manual generation using the 'save weights' button while training is active.",
    "Generate a .ckpt file when training completes.": "When enabled, a checkpoint will be generated when training completes successfully.",
    "Generate a .ckpt file when training is cancelled.": "When enabled, a checkpoint will be generated when training is cancelled by the user.",
    "Generate lora weights Generate lora weights for additional networks.": "When enabled, a lora .safetensors file will be generated in the ui lora model directory that is compatible with additional networks. Not compatible with extended lora.",
    "Generate lora weights when saving during training.": "When enabled, lora .pt files will be generated at each specified epoch interval during training. This also affects whether .pt files will be generated when manually clicking the 'Save Weights' button.",
    "Generate lora weights when training completes.": "When enabled, lora .pt files will be generated when training completes.",
    "Generate lora weights when training is canceled.": "When enabled, lora .pt files will be generated when training is cancelled by the user.",
    "Gradient Accumulation Steps": "Number of updates steps to accumulate before performing a backward/update pass.",
    "Gradient Checkpointing": "This is a technique to reduce memory usage by clearing activations of certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time for reduced memory usage.",
    "Graph Smoothing Steps": "How many timesteps to smooth graph data over. A lower value means a more jagged graph with more information, higher value will make things prettier but slightly less accurate.",
    "Growth Rate": "Prevent the D estimate from growing faster than this multiplicative rate",
    "Half Model": "Enable this to generate model with fp16 precision. Results in a smaller checkpoint with minimal loss in quality.",
    "HuggingFace Token": "Your huggingface token to use for cloning files.",
    "Instance Prompt": "A prompt describing the subject. Use [Filewords] to parse image filename/.txt to insert existing prompt here.",
    "Instance Token": "When using [filewords], this is the instance identifier that is unique to your subject. Should be a single word.",
    "Learning Rate Scheduler": "The learning rate scheduler to use. All schedulers use the provided warmup time except for 'constant'.",
    "Learning Rate Warmup Steps": "Number of steps for the warmup in the lr scheduler. LR will start at 0 and increase to this value over the specified number of steps.",
    "Learning Rate": "The rate at which the model learns. Default is 2e-6.",
    "Load Settings": "Load last saved training parameters for the model.",
    "Log Memory": "Log the current GPU memory usage.",
    "Lora Model": "The Lora model to load for continued fine-tuning or checkpoint generation.",
    "Use Lora Extended": "Trains the Lora model with resnet layers. This will always improves quality and editability, but leads to bigger files.",
    "Lora UNET Rank": "The rank for the Lora UNET (Default 4). Higher values = better quality with large file size. Lower values = sacrifice quality with lower file size. Learning rates work differently at different ranks. Saved loras at high precision (fp32) will lead to larger lora files.",
    "Lora Text Encoder Rank": "The rank for the Lora Text Encoder (Default 4). Higher values = better quality with large file size. Lower values = sacrifice quality with lower file size. Learning rates work differently at different ranks. Saved loras at high precision (fp32) will lead to larger lora files.",
    "Lora Text Learning Rate": "The learning rate at which to train lora text encoder. Regular learning rate is ignored.",
    "Lora Text Weight": "What percentage of the lora weights should be applied to the text encoder when creating a checkpoint.",
    "Lora UNET Learning Rate": "The learning rate at which to train lora unet. Regular learning rate is ignored.",
    "Lora Weight": "What percentage of the lora weights should be applied to the unet when creating a checkpoint.",
    "Max Resolution": "The resolution of input images. When using bucketing, this is the maximum size of image buckets.",
    "Max Token Length": "Maximum token length to respect. You probably want to leave this at 75.",
    "Memory Attention": "The type of memory attention to use. 'Xformers' will provide better performance than flash_attention, but requires a separate installation.",
    "Min Learning Rate": "The minimum learning rate to decrease to over time.",
    "Mixed Precision": "Use FP16 or BF16 (if available) will help improve memory performance. Required when using 'xformers'.",
    "Model Path": "The URL to the model on huggingface. Should be in the format of 'developer/model_name'.",
    "Model": "The model to train.",
    "Momentum": "The momentum to use for the optimizer.",
    "NoProx": "How to perform the decoupled weight decay.",
    "Name": "The name of the model to create.",
    "Number of Hard Resets": "Number of hard resets of the lr in cosine_with_restarts scheduler.",
    "Number of Samples to Generate": "How many samples to generate per subject.",
    "Offset Noise": "Allows the model to learn brightness and contrast with greater detail during training. Value controls the strength of the effect, 0 disables it.",
    "Pad Tokens": "Pad the input images token length to this amount. You probably want to do this.",
    "Pause After N Epochs": "Number of epochs after which training will be paused for the specified time. Useful if you want to give your GPU a rest.",
    "Performance Wizard (WIP)": "Attempt to automatically set training parameters based on total VRAM. Still under development.",
    "Polynomial Power": "Power factor of the polynomial scheduler.",
    "Pretrained VAE Name or Path": "To use an alternate VAE, you can specify the path to a directory containing a pytorch_model.bin representing your VAE.",
    "Preview Prompts": "Generate a JSON representation of prompt data used for training.",
    "Prior Loss Weight": "Prior loss weight.",
    "Sample CFG Scale": "The Classifier-Free Guidance Scale to use for preview images.",
    "Sample Image Prompt": "The prompt to use when generating preview images.",
    "Sample Negative Prompt": "A negative prompt to use when generating preview images.",
    "Sample Prompt Template File": "The path to a txt file to use for sample prompts. Use [filewords] or [name] to insert class token in sample prompts",
    "Sample Prompt": "The prompt to use to generate a sample image",
    "Sample Seed": "The seed to use when generating samples. Set to -1 to use a random seed every time.",
    "Sample Steps": "The number of steps to use when generating classifier/regularization images.",
    "Sanity Sample Prompt": "A prompt used to generate a 'baseline' image that will be created with other samples to verify model fidelity.",
    "Sanity Sample Seed": "The seed to use when generating the validation sample image. -1 is not supported.",
    "Save Checkpoint to Subdirectory": "When enabled, checkpoints will be saved to a subdirectory in the selected checkpoints folder.",
    "Save Model Frequency (Epochs)": "Save a checkpoint every N epochs.",
    "Save Model Frequency (Step)": "Save a checkpoint every N epochs. Must be divisible by batch number.",
    "Save Preview(s) Frequency (Epochs)": "Generate preview images every N epochs.",
    "Save Preview(s) Frequency (Step)": "Generate preview images every N steps. Must be divisible by batch number.",
    "Save Settings": "Save the current training parameters to the model config file.",
    "Save Weights": "Save weights/checkpoint/snapshot as specified in the saving section for saving 'during' training.",
    "Save and Test Webhook": "Save the currently entered webhook URL and send a test message to it.",
    "Save separate diffusers snapshots when saving during training.": "When enabled, a unique snapshot of the diffusion weights will be saved at each specified epoch interval. This uses more HDD space (A LOT), but allows resuming from training, including the optimizer state.",
    "Save separate diffusers snapshots when training completes.": "When enabled, a unique snapshot of the diffusion weights will be saved when training completes. This uses more HDD space, but allows resuming from training including the optimizer state.",
    "Save separate diffusers snapshots when training is cancelled.": "When enabled, a unique snapshot of the diffusion weights will be saved when training is canceled. This uses more HDD space, but allows resuming from training including the optimizer state.",
    "Save EMA Weights to Generated Models": "If a model was extracted or trained with EMA weights, these will be appended separately to the model for use in training later.",
    "Scale Position": "The percent in training where the 'final' learning rate should be achieved. If training at 100 epochs and this is set to 0.25, the final LR will be reached at epoch 25.",
    "Image Generation Scheduler": "Model scheduler to use. Only applies to models before 2.0.",
    "Set Gradients to None When Zeroing": "When performing the backwards pass, gradients will be set to none, instead of creating a new empty tensor. This will slightly improve VRAM.",
    "Shuffle After Epoch": "When enabled, will shuffle the dataset after the first epoch. Will enable text encoder training and latent caching (More VRAM).",
    "Shuffle Tags": "When enabled, tags after the first ',' in a prompt will be randomly ordered, which can potentially improve training.",
    "Source Checkpoint": "The source checkpoint to extract for training.",
    "Step Ratio of Text Encoder Training": "The number of steps per image (Epoch) to train the text encoder for. Set 0.5 for 50% of the epochs",
    "Strict Tokens": "Parses instance prompts separated by the following characters [,;.!?], and prevents breaking up tokens when using the tokenizer. Useful if you have prompts separated by a lot of tags.",
    "Total Number of Class/Reg Images": "Total number of classification/regularization images to use. If no images exist, they will be generated. Set to 0 to disable prior preservation.",
    "Train Imagic Only": "Uses Imagic for training instead of full dreambooth, useful for training with a single instance image.",
    "Train Text Encoder": "Enabling this will provide better results and editability, but cost more VRAM.",
    "Train": "Start training.",
    "Training Steps Per Image (Epochs)": "This is the total number of training steps that will be performed on each instance image.",
    "Training Wizard (Object/Style)": "Calculate training parameters for a non-human subject based on number of instance images and set larning rate. Disables prior preservation.",
    "Training Wizard (Person)": "Calculate training parameters for a human subject based on number of instance images and set learning rate. Enables prior preservation.",
    "Unfreeze Model": "Unfreezes model layers and allows for potentially better training, but makes increased VRAM usage more likely.",
    "Use 8bit Adam": "Enable this to save VRAM.",
    "Use CPU Only (SLOW)": "Guess what - this will be incredibly slow, but it will work for < 8GB GPUs.",
    "Use Concepts List": "Train multiple concepts from a JSON file or string.",
    "Use EMA": "Enabling this will provide better results and editability, but cost more VRAM.",
    "Use EMA Weights for Inference": "Enabling this will save the EMA unet weights as the 'normal' model weights and ignore the regular unet weights.",
    "Use Epoch Values for Save Frequency": "When enabled, save frequencies below are based on number of epochs. When disabled, frequencies are based on number of training steps.",
    "Use LORA": "Uses Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning. Uses less VRAM, saves a .pt file instead of a full checkpoint",
    "Use Lifetime Epochs When Saving": "When checked, will save preview images and checkpoints using lifetime epochs, versus current training epochs.",
    "Use Lifetime Steps When Saving": "When checked, will save preview images and checkpoints using lifetime steps, versus current training steps.",
}

// Do a thing when the UI updates
onUiUpdate(function () {
    let db_active = document.getElementById("db_active");
    if (db_active) {
        db_active.parentElement.style.display = "none";
    }

    let cm = getRealElement("change_modal");
    let cl = getRealElement("change_log");
    if (cm && cl) {
        if (cl.innerHTML !== "" && modalShown !== true) {
            modalShown = true;
            cm.classList.add("active");
        }
    }

    let errors = getRealElement("launch_errors");
    if (errors !== null && errors !== undefined && !locked && errors.innerHTML !== "") {
        let hr = getRealElement("hint_row");
        hr.innerHTML = errors.innerHTML;
        toggleComponents(false, true);
    }

    if (closeBtn === null || closeBtn === undefined) {
        let cb = getRealElement("close_modal");
        closeBtn = cb;
        if (cb && cm) {
            toggleComponents(false, false);
            cb.addEventListener("click", function () {
                cm.classList.remove("active");
            });
        }
    }

    db_progressbar();

    gradioApp().querySelectorAll('span, button, select, p').forEach(function (span) {
        let tooltip = db_titles[span.textContent];
        if (span.disabled || span.classList.contains(".\\!cursor-not-allowed")) {
            tooltip = "Select or Create a Model."
        }

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
        btn.onchange = function () {
            // Dummy function, so we don't keep setting up the observer.
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
    try {
        handleNumberInputs();
    } catch (e) {
        console.log("Gotcha: ", e);
    }

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
let mutationObserver = null;

function db_progressbar() {
    // gradio 3.8's enlightened approach allows them to create two nested div elements inside each other with same id
    // every time you use gr.HTML(elem_id='xxx'), so we handle this here
    let progressbar = gradioApp().querySelector("#db_progressbar #db_progressbar");
    let progressbarParent;
    if (progressbar) {
        progressbarParent = gradioApp().querySelector("#db_progressbar");
    } else {
        progressbar = gradioApp().getElementById("db_progressbar");
        progressbarParent = null;
    }

    let galleryButtons = gradioApp().querySelectorAll('#db_gallery .gallery-item');
    let gallery = gradioApp().getElementById("db_gallery");

    if (gallery !== null && gallery !== undefined) {
        if (galleryButtons.length !== 0) {
            gallery.style.display = "block";
        } else {
            gallery.style.display = "none !important";
        }
    }
    // let skip = id_skip ? gradioApp().getElementById(id_skip) : null;
    let interrupt = gradioApp().getElementById("db_cancel");
    let gen_sample = gradioApp().getElementById("db_train_sample");
    let gen_ckpt = gradioApp().getElementById("db_gen_ckpt");
    let gen_ckpt_during = gradioApp().getElementById("db_gen_ckpt_during")
    let train = gradioApp().getElementById("db_train");

    if (progressbar && progressbar.offsetParent) {
        if (progressbar.innerText) {
            let newtitle = '[' + progressbar.innerText.trim() + '] Stable Diffusion';
            if (document.title !== newtitle) {
                document.title = newtitle;
            }
        } else {
            let newtitle = 'Stable Diffusion'
            if (document.title !== newtitle) {
                document.title = newtitle;
            }
        }
    }

    if (progressbar != null) {
        mutationObserver = mutationObserver || new MutationObserver(function (m) {
            if (progressTimeout) {
                return;
            }

            let progress_indicator = gradioApp().querySelector("#db_active input[type='checkbox']");
            let is_active = progress_indicator.checked;

            let progressDiv = gradioApp().querySelector(".progressDiv");
            if (progressbarParent && progressDiv) {
                progressbar.style.width = progressbarParent.clientWidth + "px";
                progressDiv.style.width = progressbarParent.clientWidth + "px";
            }

            let preview = gradioApp().getElementById("db_preview");
            let gallery = gradioApp().getElementById("db_gallery");

            if (preview != null && gallery != null) {
                preview.style.width = gallery.clientWidth + "px"
                preview.style.height = gallery.clientHeight + "px"

                //only watch gallery if there is a generation process going on
                checkDbGallery();

                if (is_active) {
                    progressTimeout = window.setTimeout(function () {
                        requestMoreDbProgress();
                    }, 500);
                } else {
                    training_started = false;
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
        mutationObserver.observe(progressbar, { childList: true, subtree: true });
    }
}

function checkDbGallery() {
    if (gallerySet) return;
    let gallery = gradioApp().getElementById("db_gallery");
    // if gallery has no change, no need to setting up observer again.
    if (gallery) {
        if (galleryObserver) {
            galleryObserver.disconnect();
        }
        // Get the last selected item in the gallery.
        let prevSelectedIndex = selected_gallery_index();

        // Make things clickable?
        galleryObserver = galleryObserver || new MutationObserver(function () {
            let galleryButtons = gradioApp().querySelectorAll('#db_gallery .gallery-item');
            let galleryBtnSelected = gradioApp().querySelector('#db_gallery .gallery-item.\\!ring-2');
            let gallery = gradioApp().getElementById("db_gallery");
            if (galleryButtons.length !== 0) {
                gallery.style.display = "block";
            } else {
                gallery.style.display = "none !important";
            }

            if (prevSelectedIndex !== -1 && galleryButtons.length > prevSelectedIndex && !galleryBtnSelected) {
                // automatically re-open previously selected index (if exists)
                let activeElement = gradioApp().activeElement;
                let scrollX = window.scrollX;
                let scrollY = window.scrollY;

                galleryButtons[prevSelectedIndex].click();

                // When the gallery button is clicked, it gains focus and scrolls itself into view
                // We need to scroll back to the previous position
                setTimeout(function () {
                    window.scrollTo(scrollX, scrollY);
                }, 50);

                if (activeElement) {
                    setTimeout(function () {
                        activeElement.focus({
                            preventScroll: true // Refocus the element that was focused before the gallery was opened without scrolling to it
                        })
                    }, 1);
                }
            }
        })
        galleryObserver.observe(gallery, { childList: true, subtree: false });
        gallerySet = true;

    }
}

function requestDbProgress() {
    let btn = gradioApp().getElementById("db_check_progress_initial");
    if (btn == null) {
        console.log("Can't find da button!.")
        return;
    }
    btn.click();
    db_progressbar();
}

function requestMoreDbProgress() {
    let btn = gradioApp().getElementById("db_check_progress");
    if (btn == null) {
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
    if (progressDiv && interrupt && train && gen_sample) {
        if (training_started) {
            gen_sample.style.display = "block";
            train.style.display = "none";
            interrupt.style.display = "block";
            gen_ckpt.style.display = "none";
            gen_ckpt_during.style.display = "block";
        } else {
            train.style.display = "none";
            interrupt.style.display = "block";
            gen_ckpt.style.display = "none";
        }
    }
}
