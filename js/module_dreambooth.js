let uiMode = "normal"; // Can be one of training, normal, or creating
let trainMode = "Default";
let trainLora = false;
let trainEma = false;
let lastConcept = -1;
let conceptsList = [];
let dbListenersLoaded = false;
let dbDefaults;
const dbModule = new Module("Training", "moduleDreambooth", "moon", false, 2, initDreambooth, refreshDreambooth);

function initDreambooth() {
    sendMessage("get_db_vars", {}, true).then(function (response) {
        console.log("Got DB vars: ", response);
        createElements(response["defaults"]);
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

        let pg = document.getElementById("dreamProgress");
        let dreamProgress = new ProgressGroup(document.getElementById("dreamProgress"), prog_opts);
        $("#dreamModelSelect").modelSelect();
        $("#db_new_model").modelSelect();
        dreamProgress.setOnCancel(onDbEnd);
        dreamProgress.setOnComplete(onDbEnd);
        dreamProgress.setOnStart(onDbStart);
        dreamProgress.setOnUpdate(onDbUpdate);

        // Gallery creation. Options can also be passed to .update()
        let dreamGallery = new InlineGallery(document.getElementById('dreamGallery'), gallery_opts);
        loadDbListeners();
    });


}

function onDbEnd() {
}

function onDbStart() {
}

function onDbUpdate() {
}

function refreshDreambooth() {

}

function createElements(defaults) {
    let elementGroups = {};
    dbDefaults = defaults;
    // Enumerate key/values in defaults
    for (let key in defaults) {
        let value = defaults[key];
        let group = value.hasOwnProperty("group") ? value["group"] : "NOGROUP";
        if (!elementGroups.hasOwnProperty(group)) {
            elementGroups[group] = [];
        }
        value["key"] = key;
        elementGroups[group].push(value);
    }

    for (let group in elementGroups) {
        let groupElements = elementGroups[group];

        let groupTitle = group.replace(" ", "_");
        let groupBody = document.getElementById("db" + groupTitle);
        let groupWrap = false;
        let groupDiv = false;
        console.log("Group: ", group, groupTitle, groupBody);
        if (groupBody === null) {
            console.log("Creating group: ", groupTitle);
            groupWrap = $("<div>", {class: "accordion-item dbAccordion", id: "accordionWrap" + groupTitle});
            let groupHeader = $("<h2>", {class: "accordion-header", id: "accordionHeader" + groupTitle});
            let groupBtn = $("<button>", {
                class: "accordion-button collapsed",
                type: "button",
                'data-bs-toggle': "collapse",
                'data-bs-target': "#accordionGroup" + groupTitle,
                'aria-expanded': "false",
                'aria-controls': "accordionGroup" + groupTitle
            }).text(group);
            let groupDiv = $("<div>", {
                class: "accordion-collapse collapse",
                id: "accordionGroup" + groupTitle,
                'aria-labelledby': "accordionHeader" + groupTitle,
                'data-bs-parent': "#dbSettingsContainer"
            });
            groupBody = $("<div>", {class: "row pt-2", id: "db" + groupTitle});
            groupHeader.append(groupBtn);
            groupDiv.append(groupBody);
            groupWrap.append(groupHeader);
            groupWrap.append(groupDiv);
            $("#dbSettingsContainer").append(groupWrap);
        }

        for (let i = 0; i < groupElements.length; i++) {
            let element = groupElements[i];
            if (element.key === "concepts_list") {
                let conceptContainer = createConceptContainer();
                groupBody.append(conceptContainer);
            } else {
                let elementDiv = createElement(element, "db", ["dbInput"]);
                if (elementDiv !== null) {
                    elementDiv.classList.add("col-12", "db-group");
                    if (groupElements.length > 1) {
                        elementDiv.classList.add("col-md-6");
                    }
                    groupBody.append(elementDiv);
                } else {
                    // TODO: Handle model data in UI
                }
            }
        }

    }

    toggleElements();
}

function toggleElements() {
    console.log("Toggling elements, train mode is: ", trainMode);
    $(".FineTuneOnly").hide();
    $(".DefaultOnly").hide();
    $(".ControlNetOnly").hide();
    let dreamConfig = dbModule.systemConfig;
    let showAdvanced = dreamConfig["show_advanced"];
    $(".db-advanced").toggle(showAdvanced);
    $(".db-basic").toggle(!showAdvanced);
    $("." + trainMode + "Only").show();
    $(".loraOnly").toggle(trainLora);
    $(".emaOnly").toggle(trainEma);

    // For each .dbAccordion, check if all .dbInput inside are hidden. If so, hide the .dbAccordion.
    $(".dbAccordion").each(function () {
        let accordion = $(this);
        let allHidden = accordion.find(".db-group").filter(function () {
            return $(this).css("display") !== "none";
        }).length === 0;
        accordion.toggle(!allHidden);
    });
}


function createConceptContainer() {
    // Create elements
    let formGroupDiv = $("<div>", {class: "form-group"});
    let rowDiv = $("<div>", {class: "row justify-content-between"});
    let sectionLabelDiv = $("<div>", {class: "col-auto sectionLabel", text: "Concepts"});
    let conceptControlsDiv = $("<div>", {class: "col-auto db-advanced", id: "conceptControls"});
    let addButton = $("<button>", {type: "button", class: "btn btn-primary btn-sm", id: "db_concept_add", text: "+"});
    let removeButton = $("<button>", {
        type: "button",
        class: "btn btn-danger btn-sm hide",
        id: "db_concept_remove",
        text: "-"
    });
    let conceptsListDiv = $("<div>", {id: "db_concepts_list", class: "form-control"});

    // Structure elements
    conceptControlsDiv.append(addButton, removeButton);
    rowDiv.append(sectionLabelDiv, conceptControlsDiv);
    formGroupDiv.append(rowDiv, conceptsListDiv);

    return formGroupDiv;
}

function addConcept(concept = false) {
    let dreamConfig = dbModule.systemConfig;
    let showAdvanced = dreamConfig["show_advanced"];
    let conceptsContainer = $("#db_concepts_list");
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
                     data-bs-parent="#db_concepts_list">
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
                        <div id="${inputId}" class="db-file-browser dbInput conceptInput" data-elem_id="${inputId}"></div>
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
                        <div id="${inputId}" class="db-slider dbInput conceptInput" data-elem_id="${inputId}" data-max="${sliderMax}" data-min="${sliderMin}" data-step="${sliderStep}" data-value="${concept[key]}" data-label="${bootstrapSliderLabel}"></div>
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
                            <input type="text" class="form-control dbInput conceptInput" id="${inputId}" value="${concept[key]}">
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
                            <input type="text" class="form-control dbInput conceptInput" id="${inputId}" value="${concept[key]}">
                        </div>
                    `);

            formElements.append(text);
        } else if (numberKeys.includes(key)) {
            let numberLabel = key === "sample_seed" ? "Sample Seed" : "";
            let number = $(`
                        <div class="form-group">
                            <label for="${inputId}">${numberLabel}</label>
                            <input type="number" class="form-control dbInput conceptInput" id="${inputId}" value="${concept[key]}">
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


function loadDbListeners() {
    $("#db_train_mode").change(function () {
        trainMode = $(this).val();
        toggleElements();
    });
    $("#db_train_lora").change(function () {
        trainLora = $(this).is(":checked");
        toggleElements();
    });
    $("#db_train_ema").change(function () {
        trainEma = $(this).is(":checked");
        toggleElements();
    });
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

    $("#loadDbSetting").click(function () {
        modelLoaded = true;
        let selected = $("#dreamModelSelect").modelSelect().getModel();
        if (selected === undefined) {
            alert("Please select a model first!");
        } else {
            console.log("Fetching model data for: ", selected);
            sendMessage("get_db_config", {model: selected}, true).then((result) => {
                const trainConfig = result["config"];
                console.log("Loading settings: ", trainConfig);
                for (let key in trainConfig) {
                    let value = trainConfig[key]["value"];
                    console.log("Loading: ", key, value);
                    if (value === null || value === undefined) continue;
                    if (key === "concepts_list") {
                        loadConcepts(value);
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
                        } else if (key === "train_data_dir") {
                            ftDataDir.setCurrentPath(value);
                            ftDataDir.setValue(value);
                        } else {
                            element.val(value);
                        }
                    }

                }

            });
        }
    });

    $("#saveDbSettings").click(function () {
        let selected = $("#dreamModelSelect").modelSelect().getModel();
        if (selected === undefined) {
            alert("Please select a model first!");
        } else {
            let data = getSettings();
            console.log("Got settings:", data);
            data["pretrained_model_name_or_path"] = selected["path"];
            data["model_name"] = selected["name"];
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

    keyListener.register("ctrl+Enter", "#dreamSettings", startTraining);

    dbListenersLoaded = true;
}

function startTraining() {

}

function loadConcepts(concepts) {
    console.log("Loading concepts: ", concepts);
    let conceptsContainer = $("#db_concepts_list");
    conceptsList = [];
    conceptsContainer[0].innerHTML = "";
    if (concepts.length === 0) {
        let removeConceptButton = $("#db_concept_remove");
        let controlGroup = $("#conceptControls");
        controlGroup.removeClass("btn-group");
        removeConceptButton.addClass("hide");
        removeConceptButton.hide();
        return;
    }


    for (let i = 0; i < concepts.length; i++) {
        let concept = concepts[i];
        addConcept(concept);
    }

}


function updateUiButtons() {

}

function updateUi() {

}

function getConcepts() {
    let concepts = [];
    // Find all the elements in the container who's
    let conceptElements = $(".conceptInput");
    let outputs = {};
    console.log("Concept elements: ", conceptElements);
    for (let i = 0; i < conceptElements.length; i++) {
        let conceptElement = conceptElements[i];
        let elementId = conceptElement.id;
        let parts = elementId.split("-");
        let keyName = parts[1]
        let conceptKey = parts[0].replace("concept_", "");
        let value = getElementValue(elementId);
        if (value !== null) {
            if (outputs[conceptKey] === undefined) {
                outputs[conceptKey] = {};
            }
            outputs[conceptKey][keyName] = value;
        }
    }
    // Convert the dictionary to a list
    for (let key in outputs) {
        concepts.push(outputs[key]);
    }
    return concepts;
}

function getSettings() {
    let selected = $("#dreamModelSelect").modelSelect().getModel();
    console.log("Selected: ", selected);
    let params = {};
    if (selected !== undefined && selected !== null) {
        params["pretrained_model_name_or_path"] = selected["path"];
        params["model_name"] = selected["name"];
    }
    params["model"] = selected;
    let elementValue;
    console.log("Getting db params: ", dbDefaults);
    for (let key in dbDefaults) {
        let dbObj = dbDefaults[key];
        if (dbObj.hasOwnProperty("description")) {
            if (dbObj.description.indexOf("[model]") !== -1) {
                continue;
            }
        }
        let lookKey = key;
        if (key === "concepts_list") {
            elementValue = getConcepts();
        } else {
            let element = $("#db_" + lookKey);
            if (element.length === 0) {
                console.log("Could not find element(0) with id: ", "db_" + lookKey);
            } else {
                elementValue = getElementValue("db_" + lookKey);
            }
        }
        if (elementValue !== null) {
            params[key] = elementValue;
        }
    }
    return params;
}