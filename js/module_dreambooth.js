let dreamSelect;
let dreamConfig;
let showAdvanced;
let dreamProgress;
let dreamGallery;
let lastConcept = -1;
let conceptsList = [];
let dbListenersLoaded = false;
let dbModelCol;
let dbStatusCol;
let linkLR = false;
let modelLoaded = false;

// Define the Bootstrap "md" breakpoint as a constant
const mdBreakpoint = 990;

// Register the module with the UI. Icon is from boxicons by default.
const dbModule = new Module("Dreambooth", "moduleDreambooth", "moon", false, 2, initDreambooth, refreshDreambooth);

function initDreambooth() {
    sendMessage("get_db_vars", {}, true).then(function (response) {
        console.log("Got DB vars: ", response);
        let attentionSelect = $("#attention");
        let precisionSelect = $("#mixed_precision");
        let schedulerSelect = $("#lr_scheduler");
        let optimizerSelect = $("#optimizer");
        let imgSchedulerSelect = $("#scheduler");

        // iterate over each array in the response and populate the corresponding select object
        for (let key in response) {
            let arr = response[key];
            let select = null;
            switch (key) {
                case "attentions":
                    select = attentionSelect;
                    break;
                case "precisions":
                    select = precisionSelect;
                    break;
                case "schedulers":
                    select = schedulerSelect;
                    break;
                case "optimizers":
                    select = optimizerSelect;
                    break;
                case "infer_schedulers":
                    select = imgSchedulerSelect;
                default:
                    break;
            }
            if (select) {
                // populate the select object with options
                for (let i = 0; i < arr.length; i++) {
                    let value = arr[i];
                    let displayValue = value.replace(/_/g, " ").toTitleCase();
                    if (key === "infer_schedulers") displayValue = value;
                    let option = $("<option></option>").attr("value", value).text(displayValue);
                    select.append(option);
                }
                // set the default selected option
                let defaultValue = null;
                switch (key) {
                    case "attentions":
                        if (arr.includes("xformers")) {
                            defaultValue = "xformers";
                        } else {
                            defaultValue = "default";
                        }
                        break;
                    case "precisions":
                        if (arr.includes("bf16")) {
                            defaultValue = "bf16";
                        } else {
                            defaultValue = "fp16";
                        }
                        break;
                    case "schedulers":
                        defaultValue = "constant_with_warmup";
                        break;
                    case "optimizers":
                        if (arr.includes("8bit AdamW")) {
                            defaultValue = "8bit AdamW";
                        } else if (arr.includes("Torch AdamW")) {
                            defaultValue = "Torch AdamW";
                        }
                        break;
                    case "infer_schedulers":
                        defaultValue = "UniPCMultistep";
                    default:
                        break;
                }
                if (defaultValue) {
                    let defaultOption = select.find(`option[value="${defaultValue}"]`);
                    if (defaultOption.length > 0) {
                        defaultOption.attr("selected", true);
                    } else {
                        select.find("option:first").attr("selected", true);
                    }
                } else {
                    select.find("option:first").attr("selected", true);
                }
            }
        }
    });

    // utility function to convert a string to Title Case
    String.prototype.toTitleCase = function () {
        return this.replace(/\w\S*/g, function (txt) {
            return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
        });
    };

    $("#db_new_model").modelSelect();
    dreamSelect = $("#dreamModelSelect").modelSelect();
    dreamSelect.setOnChangeHandler(function (value) {
        let md = dreamSelect.getModel();
        console.log("DS change: ", md, value);
        if (md && md.hasOwnProperty("data")) {
            let data = md.data;
            if (data.hasOwnProperty("db_config")) {
                setModelInfo(data.db_config);
            }
        }
    });

    let prog_opts = {
        "primary_status": "Status 1", // Status 1 text
        "secondary_status": "Status 2", // Status 2...
        "bar1_progress": 0, // Progressbar 1 position
        "bar2_progress": 0, // etc
        "id": "dreamProgress" // ID of the progress group
    }

    let gallery_opts = {
        "thumbnail": true,
        "closeable": false,
        "show_maximize": true,
        "start_open": true,
        "id": "dreamProgress"
    }

    window.addEventListener('resize', handleResize);

    // Call the function once on page load to ensure the correct container is shown
    handleResize();
    let pg = document.getElementById("dreamProgress");
    dreamProgress = new ProgressGroup(document.getElementById("dreamProgress"), prog_opts);

    dreamProgress.setOnCancel(onDbEnd);
    dreamProgress.setOnComplete(onDbEnd);
    dreamProgress.setOnStart(onDbStart);
    dreamProgress.setOnUpdate(onDbUpdate)

    // Gallery creation. Options can also be passed to .update()
    dreamGallery = new InlineGallery(document.getElementById('dreamGallery'), gallery_opts);


    $(".db-slider").BootstrapSlider();

    if (!dbListenersLoaded) {
        loadDbListeners();
    }

    registerSocketMethod("train_dreambooth", "train_dreambooth", dreamResponse);

    dreamConfig = dbModule.systemConfig;
    showAdvanced = dreamConfig["show_advanced"];
    if (showAdvanced) {
        $(".db-advanced").show();
        $(".db-basic").hide();
        $("#hub_row").hide();
        $("#local_row").show();
    } else {
        $(".db-advanced").hide();
        $(".db-basic").show();
    }
}

function refreshDreambooth() {
    dreamConfig = dbModule.systemConfig;
    showAdvanced = dreamConfig["show_advanced"];
    if (showAdvanced) {
        $(".db-advanced").show();
        $(".db-basic").hide();
        $("#hub_row").hide();
        $("#local_row").show();
    } else {
        $(".db-advanced").hide();
        $(".db-basic").show();
    }
}
function onDbEnd() {
    $(".dbTrainBtn").addClass("hide");
    $(".dbSettingBtn").removeClass("hide");
}

function onDbStart() {
    $(".dbTrainBtn").removeClass("hide");
    $(".dbSettingBtn").addClass("hide");
}

function onDbUpdate(data) {
    console.log("onDbUpdate", data);
    // If the data is a JSON string, parse it
    let status = data["status"];
    if (status.hasOwnProperty("status")) {
        status = status["status"];
    }
    console.log("OG status: ", status);

    if (typeof status === "string" && status.startsWith("{")) {
        let stats = JSON.parse(status);
        console.log("Stats:", stats);
        const progress_1_total = stats.total_session_step;
        const progress_1_current = stats.session_step;
        const progress_2_total = stats.steps_per_epoch;
        const progress_2_current = stats.session_step - (stats.steps_per_epoch * stats.session_epoch);
        // Truncate loss to 2 decimal places
        let progressContainer = $("#dreamProgress").find(".progressContainer");
        // If there is no .plotter class in the progressContainer, add one
        // Check if the plotter element already exists
        if (progressContainer.find(".plotter").length === 0) {
            // Create the plotter element if it doesn't exist
            progressContainer.append(`<div class="plotter" id="dbChart"></div>`);
            let plotter = document.getElementById("dbChart");

            // Initialize the plot with the initial data for loss, VRAM usage, and learning rate
            let data = [
                {
                    y: [stats.loss],
                    x: [stats.session_step],
                    mode: 'lines',
                    name: 'Loss',
                    yaxis: 'y'
                },
                {
                    y: [stats.unet_lr],
                    x: [stats.session_step],
                    mode: 'lines',
                    name: 'LR',
                    yaxis: 'y2'
                }
            ];

            // Specify the layout options for the plot
            let lr = parseFloat(stats.lr_data[0]);
            console.log("LR: ", lr);
            let layout = {
                autosize: true,
                height: 300,
                paper_bgcolor: '#7f7f7f',
                plot_bgcolor: '#c7c7c7',
                yaxis: {
                    title: 'Loss',
                    side: 'left',
                    range: [0, 1]
                },
                yaxis2: {
                    title: 'LR',
                    side: 'right',
                    overlaying: 'y',
                    range: [0, lr * 1.2]
                },
                xaxis: {
                    title: 'Steps'
                }

            };

            // Create the plot using Plotly.newPlot
            Plotly.newPlot(plotter, data, layout);
        } else {
            let plotter = document.getElementById("dbChart");

            // Retrieve the existing data from the plot
            let existingData = plotter.data;

            // Extend the existing traces by adding new points for loss, VRAM usage, and learning rate
            let lr = parseFloat(stats.lr_data[0]);
            console.log("LR: ", lr);
            Plotly.extendTraces(plotter, {
                y: [[stats.loss], [lr]],
                x: [[stats.session_step], [stats.session_step]]
            }, existingData.map((_, i) => i)); // Update all the existing traces
        }


        data = {
            "status": `Steps: ${progress_1_current}/${progress_1_total} (${stats.iterations_per_second})`,
            "status_2": `Epoch: ${stats.session_epoch}/${stats.total_session_epoch} VRAM: ${stats.vram}`,
            "progress_1_total": progress_1_total,
            "progress_1_current": progress_1_current,
            "progress_2_total": progress_2_total,
            "progress_2_current": progress_2_current,
            "target": "dreamProgress"
        };

    }
    console.log("Returning: ", data);
    return data;
}


function setModelInfo(data) {
    const keysToSelect = ["epoch", "v2", "src", "has_ema"];
    const keyTitles = {
        "epoch": "Model Epoch",
        "v2": "V2 Model",
        "src": "Source Model",
        "has_ema": "Has EMA"
    }
    const dbModelInfoElement = document.querySelector(".dbModelInfo");
    dbModelInfoElement.innerHTML = "";

    keysToSelect.forEach((key) => {
        let value = data[key];
        if (value === "false" || value === false) value = "False";
        // Create a div element for the key and value
        const divElement = document.createElement("div");
        divElement.classList.add("m-2", "d-inline-block", "col-12");

        // Create a span element for the title
        const titleSpan = document.createElement("span");
        titleSpan.textContent = keyTitles[key] + ": ";

        // Create a span element for the value
        const valueSpan = document.createElement("span");
        valueSpan.textContent = value;
        valueSpan.classList.add("fit");

        // Append the title and value spans to the div element
        divElement.appendChild(titleSpan);
        divElement.appendChild(valueSpan);

        // Append the div element to the dbModelInfoElement
        dbModelInfoElement.appendChild(divElement);
    });


    fitty(".fit",
        {
            minSize: 10,
            maxSize: 16,
            multiLine: false,
        });
    if (!modelLoaded) $("#db_load_params").click();
}

function handleResize() {
    // Get the current window width
    const windowWidth = window.innerWidth;
    dbModelCol = document.getElementById("dbModelCol");
    dbStatusCol = document.getElementById("dbStatusCol");

    let statusCard = document.getElementById("dreamStatus");
    // If the window is less than the "md" breakpoint
    if (windowWidth <= mdBreakpoint) {
        // Check if statusCard is not already a child of dbModelCol
        if (dbModelCol.contains(statusCard) === false) {
            // Append statusCard to dbModelCol
            dbModelCol.prepend(statusCard);
            //dbModelCol.appendChild(statusCard);
        }
        // Hide dbStatusCol
        dbStatusCol.style.display = "none";
    } else {
        // Check if statusCard is not already a child of dbStatusCol
        if (dbStatusCol.contains(statusCard) === false) {
            // Append statusCard to dbStatusCol
            dbStatusCol.appendChild(statusCard);
        }
        // Show dbStatusCol
        dbStatusCol.style.display = "block";
    }
}

function loadDbListeners() {
    $("#db_use_shared_src").click(function () {
        let checked = $(this).is(":checked");
        if (checked) {
            $("#shared_row").show();
        } else {
            $("#shared_row").hide();
        }
    });

    $(".linkBtn").click(function () {
        console.log("Linkyclicky.");
        $(this).toggleClass("active");
        linkLR = $(this).hasClass("active");
        $("#txt_learning_rate").prop("disabled", linkLR);
    });

    $("#learning_rate").change(function () {
        if (linkLR) {
            let val = $(this).val();
            $("#txt_learning_rate").val(val);
        }
    });

    $("#db_create_model").click(function () {
        let data = {};
        $(".newModelParam").each(function (index, elem) {
            let key = $(elem).data("key");
            let val = $(elem).val();
            if ($(elem).is(":checkbox")) {
                val = $(elem).is(":checked");
            }
            if ($(elem).is(".modelSelect")) {
                let ms = $(elem).modelSelect();
                val = ms.getModel();
                console.log("Got ms: ", key, val);
            }
            if (key === "is_512") key = "512_model";
            data[key] = val;
        });
        sendMessage("create_dreambooth", data, false, "dreamProgress").then(() => {
        });
    });

    $("#db_load_settings").click(function () {
        let selected = dreamSelect.getModel();
        if (selected === undefined) {
            alert("Please select a model first!");
        } else {
            sendMessage("get_dreambooth_settings", {model: selected}, true).then((result) => {
                for (let key in result) {
                    let elem = $(`#${key}`);
                    if (elem.length) {
                        if (elem.is(":checkbox")) {
                            elem.prop("checked", result[key]);
                        } else {
                            elem.val(result[key]);
                        }
                    }
                }
            });
        }
    });

    $("#db_train").click(function () {
        let data = getSettings();
        console.log("Settings: ", data);
        $(".dbSettingBtn").addClass("hide");
        $(".dbTrainBtn").removeClass("hide");
        sendMessage("train_dreambooth", data, false, "dreamProgress").then((result) => {
            $(".dbSettingBtn").addClass("hide");
            $(".dbTrainBtn").removeClass("hide");
        });
    });

    $("#db_cancel").click(function () {
        $(".dbSettingBtn").removeClass("hide");
        $(".dbTrainBtn").addClass("hide");
    });

    $("#db_load_params").click(function () {
        modelLoaded = true;
        let selected = $("#dreamModelSelect").modelSelect().getModel();
        if (selected === undefined) {
            alert("Please select a model first!");
        } else {
            console.log("Fetching model data for: ", selected);
            sendMessage("get_db_config", {model: selected}, true).then((result) => {
                console.log("Loading settings: ", result);
                for (let key in result["config"]) {
                    let value = result["config"][key];
                    if (value === null || value === undefined) continue;
                    if (key === "concepts_list") {
                        let concepts = result["config"][key];
                        loadConcepts(concepts);
                        continue;
                    }
                    let elem_selector = "#" + key;
                    let element = $(elem_selector);
                    if (element.length !== 0) {
                        if (element.hasClass("db-slider")) {
                            let slider = element.data("BootstrapSlider");
                            if (slider) {
                                slider.setValue(value);
                            }
                        } else if (element.is(":checkbox")) {
                            element.prop("checked", value);
                        } else {
                            element.val(value);
                        }
                    }

                }

            });
        }
    });

    $("#db_save_config").click(function () {
        let selected = $("#dreamModelSelect").modelSelect().getModel();
        if (selected === undefined) {
            alert("Please select a model first!");
        } else {
            let data = getSettings();
            sendMessage("save_db_config", data, true).then((result) => {
                console.log("Res: ", result);
            });
        }
    });

    $("#db_create_from_hub").change(function () {
        if ($(this).is(":checked")) {
            $("#hub_row").show();
            $("#local_row").hide();
        } else {
            $("#hub_row").hide();
            $("#local_row").show();
        }
    });

    $("#db_concept_add").click(function () {
        addConcept(false);
    });

    $("#db_concept_remove").click(function (event) {
        event.preventDefault();
        removeConcept();
    });

    keyListener.register("ctrl+Enter", "#dreamSettings", startDreambooth);

    dbListenersLoaded = true;
}

function loadConcepts(concepts) {
    let conceptsContainer = $("#advancedConcepts");
    conceptsList = [];
    conceptsContainer[0].innerHTML = "";
    if (concepts.length === 0 && showAdvanced) {
        let removeConceptButton = $("#db_concept_remove");
        let controlGroup = $("#conceptControls");
        controlGroup.removeClass("btn-group");
        removeConceptButton.addClass("hide");
        removeConceptButton.hide();
        return;
    }

    if (showAdvanced) {
        for (let i = 0; i < concepts.length; i++) {
            let concept = concepts[i];
            addConcept(concept);
        }
    } else {
        if (concepts.length > 0) {
            let concept = concepts[0];
            console.log("Creating existing concept: ", concept);
            addConcept(concept);
        } else {
            console.log("Creating new concept");
            addConcept(false);
        }
    }
}

function addConcept(concept = false) {
    let conceptsContainer = $("#advancedConcepts");
    let removeConcept = $("#db_concept_remove");
    let controlGroup = $("#conceptControls");
    if (showAdvanced) {
        controlGroup.addClass("btn-group");
        removeConcept.removeClass("hide");
        controlGroup[0].style.display = "flex";
        removeConcept.show();
    } else {
        controlGroup.hide();
    }

    let fileBrowserKeys = ["class_data_dir", "instance_data_dir"];
    let bootstrapSliderKeys = ["class_guidance_scale", "class_infer_steps", "n_save_sample", "num_class_images_per", "save_guidance_scale", "save_infer_steps"];
    let textFieldKeys = ["class_negative_prompt", "save_sample_negative_prompt", "save_sample_prompt", "save_sample_template"];
    let textKeys = ["class_prompt", "class_token", "instance_prompt", "instance_token"];
    let numberKeys = ["sample_seed"];

    if (!concept) {
        concept = {
            "class_data_dir": "",
            "class_guidance_scale": 7.5,
            "class_infer_steps": 20,
            "class_negative_prompt": "blurry, deformed, bad",
            "class_prompt": "[filewords]",
            "class_token": "",
            "instance_data_dir": "",
            "instance_prompt": "[filewords]",
            "instance_token": "",
            "n_save_sample": 0,
            "num_class_images_per": 0,
            "sample_seed": -1,
            "save_guidance_scale": 7.5,
            "save_infer_steps": 20,
            "save_sample_negative_prompt": "blurry, deformed, bad",
            "save_sample_prompt": "[filewords]"
        };
    }

    conceptsList.push(concept);

    const c_keys = [
        "instance_data_dir",
        "class_data_dir",
        "instance_prompt",
        "class_prompt",
        "save_sample_prompt",
        "sample_template",
        "instance_token",
        "class_token",
        "num_class_images_per",
        "n_save_sample",
        "class_negative_prompt",
        "save_sample_negative_prompt",
        "class_guidance_scale",
        "save_guidance_scale",
        "class_infer_steps",
        "save_infer_steps",
        "sample_seed"
    ];

    let i = conceptsContainer.children().length + 1;

    let formAccordion = $(`<div class="accordion-item" data-index="${i}">
                <h2 class="accordion-header" id="concept-${i}">
                  <button class="accordion-button" type="button" data-bs-toggle="collapse"
                    data-bs-target="#collapse-${i}" aria-expanded="true"
                    aria-controls="collapse-${i}">Concept ${i}</button>
                </h2>
                <div id="collapse-${i}" class="accordion-collapse collapse" aria-labelledby="concept-${i}"
                     data-bs-parent="#advancedConcepts">
                  <div class="accordion-body">
                    <form id="form-${i}"></form>
                  </div>
                </div>
              </div>`);

    formAccordion.click(function (event) {
        event.preventDefault();
        let idx = $(this).data("index");
        lastConcept = idx;
    });

    formAccordion.blur(function (event) {
        event.preventDefault();
    });

    conceptsContainer.append(formAccordion);
    let formElements = $("#form-" + i);
    // create arrays for each element type

    // loop through each key in concept
    for (let key_idx in c_keys) {
        let key = c_keys[key_idx];
        let inputId = `concept_${i}-${key}`;
        // check if the key is in the fileBrowserKeys array
        if (fileBrowserKeys.includes(key)) {
            let fileBrowserLabel = key === "class_data_dir" ? "Class Data Directory" : "Instance Data Directory";
            let fileBrowser = $(`
                    <div class="form-group">
                        <label>${fileBrowserLabel}</label>
                        <div id="${inputId}" class="db-file-browser dbInput" data-elem_id="${inputId}"></div>
                    </div>
                `);
            formElements.append(fileBrowser);
            $("#" + inputId).fileBrowser({
                "dropdown": true,
                "showInfo": false,
                "showTitle": false,
                "showSelectButton": true,
                "selectedElement": concept[key]
            });
        }
        // check if the key is in the bootstrapSliderKeys array
        else if (bootstrapSliderKeys.includes(key)) {
            let bootstrapSliderLabel = key === "class_guidance_scale" ? "Class Guidance Scale" :
                key === "class_infer_steps" ? "Class Inference Steps" :
                    key === "n_save_sample" ? "N Save Samples" :
                        key === "num_class_images_per" ? "Images per Class" :
                            key === "save_guidance_scale" ? "Sample Guidance Scale" :
                                "Sample Inference Steps";
            let sliderStep = key.indexOf("scale") !== -1 ? 0.5 : 1;
            let sliderMax = key.indexOf("scale") !== -1 ? 20 : 100;
            let sliderMin = (key.indexOf("sample") !== -1 || key.indexOf("images") !== -1) ? 0 : 1;
            let sliderDiv = $(`
                    <div class="form-group db-advanced">
                        <div id="${inputId}" class="db-slider dbInput" data-elem_id="${inputId}" data-max="${sliderMax}" data-min="${sliderMin}" data-step="${sliderStep}" data-value="${concept[key]}" data-label="${bootstrapSliderLabel}"></div>
                    </div>
                `);

            formElements.append(sliderDiv);
            $("#" + inputId).BootstrapSlider({});
        }
        // check if the key is in the textFieldKeys array
        else if (textFieldKeys.includes(key)) {
            let textFieldLabel = key === "class_negative_prompt" ? "Class Negative Prompt" :
                key === "save_sample_negative_prompt" ? "Sample Negative Prompt" :
                    key === "save_sample_prompt" ? "Sample Prompt" :
                        "Sample Template";
            let textField = $(`
                        <div class="form-group db-advanced">
                            <label for="${inputId}">${textFieldLabel}</label>
                            <input type="text" class="form-control dbInput" id="${inputId}" value="${concept[key]}">
                        </div>
                    `);
            formElements.append(textField);
        } else if (textKeys.includes(key)) {
            let textLabel = key === "class_prompt" ? "Class Prompt" :
                key === "class_token" ? "Class Token" :
                    key === "instance_prompt" ? "Instance Prompt" :
                        "Instance Token";
            let text = $(`
                        <div class="form-group">
                            <label for="${inputId}">${textLabel}</label>
                            <input type="text" class="form-control dbInput" id="${inputId}" value="${concept[key]}">
                        </div>
                    `);

            formElements.append(text);
        } else if (numberKeys.includes(key)) {
            let numberLabel = key === "sample_seed" ? "Sample Seed" : "";
            let number = $(`
                        <div class="form-group">
                            <label for="${inputId}">${numberLabel}</label>
                            <input type="number" class="form-control dbInput" id="${inputId}" value="${concept[key]}">
                        </div>
                    `);
            formElements.append(number);
        }
    }
    // create bootstrap slider elements
    formElements.find(".db-slider").BootstrapSlider();
    if (showAdvanced) {
        $(".db-advanced").show();
        $(".db-basic").hide();
        $("#hub_row").hide();
        $("#local_row").show();
    } else {
        $(".db-advanced").hide();
        $(".db-basic").show();
    }
}

function removeConcept() {
    if (lastConcept === -1) {
        return;
    }
    let c_idx = lastConcept - 1;
    // Remove the element from conceptsList at index c_idx
    conceptsList.splice(c_idx, 1);
    loadConcepts(conceptsList);
}

function getSettings() {
    let settings = {};
    settings["model"] = $("#dreamModelSelect").modelSelect().getModel();

    // Just create one concept if advanced is disabled
    let concepts_list = [];

    let conceptElements = $('[id^="concept_"]');

    let to_skip = ["_container", "_range", "_number"];
    let allSettings = dbModule.getSettings(".dbInput", to_skip);
    console.log("All settings: ", allSettings);
    // Enumerate through allSettings, removing any element with "concept_" in the key and putting it in a new dict.
    let concepts = {};

    for (let key in allSettings) {
        if (key.indexOf("concept_") !== -1) {
            let conceptIdx = key.split("-")[0].split("_")[1];
            let conceptKey = key.split("-")[1];
            if (!(conceptIdx in concepts)) {
                concepts[conceptIdx] = {};
            }
            concepts[conceptIdx][conceptKey] = allSettings[key];
        } else {
            if (key === "is_512") key = "512_model";
            settings[key] = allSettings[key];
        }
    }
    // Convert concepts to a list
    concepts_list = [];
    for (let key in concepts) {
        concepts_list.push(concepts[key]);
    }
    settings["concepts_list"] = concepts_list;
    console.log("Cleaned settings: ", settings);
    if (!showAdvanced) {
        settings["txt_learning_rate"] = settings["learning_rate"] / 2;
    }
    return settings;
}


function dreamResponse() {

}

function startDreambooth() {

}
