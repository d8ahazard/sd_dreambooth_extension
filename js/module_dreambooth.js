let inputBrowser;
let dreamSelect;
let dreamConfig;

$(".hide").hide();
document.addEventListener("DOMContentLoaded", function () {
    let selects = $(".modelSelect").modelSelect();
    console.log("Selects: ", selects);
    for (let i = 0; i < selects.length; i++) {
        let elem = selects[i];
        console.log("ELEM: ", elem);
        if (elem.container.id === "dreamModelSelect") {
            dreamSelect = elem;
        }
    }
    console.log("DS: ", dreamSelect);
    $("#db_create_model").click(function () {
        let data = {};
        $(".newModelParam").each(function (index, elem) {
            let key = $(elem).data("key");
            let val = $(elem).val();
            console.log("ELEM: ", elem);
            if ($(elem).is(":checkbox")) {
                val = $(elem).is(":checked");
            }
            if ($(elem).is(".model-select")) {
                let parent = $(elem).parent().parent();
                let ms = $(parent).modelSelect();
                console.log("MS: ", ms);
                let realElement = ms[0];
                console.log("RE: ", realElement);
                val = realElement.currentModel;
                console.log("MS: ", val);

            }
            data[key] = val;
        });
        sendMessage("create_dreambooth", data, true).then(() => {
            console.log("New model params: ", data);
            dreamSelect.refresh();
        });

    });


    $("#db_load_settings").click(function () {
        let selected = dreamSelect.getModel();
        console.log("Load model settings click: ", selected);
        if (selected === undefined) {
            alert("Please select a model first!");
        } else {
            sendMessage("get_dreambooth_settings", {model: selected}, true).then((result) => {
                console.log("Settings: ", result);
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
        sendMessage("train_dreambooth", data, false).then((result) => {
            console.log("Res: ", result);
        });
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

    inputBrowser = new FileBrowser(document.getElementById("dreamInputBrowser"), {
        "dropdown": true,
        "showInfo": false,
        "showTitle": false,
        "showSelectButton": true
    });

    inferProgress = new ProgressGroup(document.getElementById("dreamProgress"), {
        "primary_status": "Status 1", // Status 1 text
        "secondary_status": "Status 2", // Status 2...
        "bar1_progress": 0, // Progressbar 1 position
        "bar2_progress": 0 // etc
    });

    // Gallery creation. Options can also be passed to .update()
    gallery = new InlineGallery(document.getElementById('dreamGallery'), {
        "thumbnail": true,
        "closeable": false,
        "show_maximize": true,
        "start_open": true
    });

    $(".db-slider").BootstrapSlider();

    // Register the module with the UI. Icon is from boxicons by default.
    registerModule("Dreambooth", "moduleDreambooth", "cloud", false, 2);


    registerSocketMethod("train_dreambooth", "train_dreambooth", dreamResponse);


    keyListener.register("ctrl+Enter", "#dreamSettings", startDreambooth);

    sendMessage("get_layout", {}).then((data) => {
       console.log("LAYOUT DATA: ", data);
    });

    sendMessage("get_config", {"section_key": "dreambooth"}).then((data) => {
        dreamConfig = data;
        console.log("Dream settings: ", dreamConfig);
        if (dreamConfig["show_advanced"]) {
            $(".db-advanced").show();
            $(".db-basic").hide();
        }
    });
});


function getSettings() {
    let concept_path = inputBrowser.currentPath;
    let dbInputs = $(".db-slider");
    let settings = {};
    settings["model"] = dreamSelect.getModel();
    settings["c1_instance_data_dir"] = concept_path;
    dbInputs.each(function () {
        let element = $(this);
        console.log("ELEM: ", element);
        let id = element.data("elem_id");
        let slider = element.data("BootstrapSlider");
        console.log("Slider: ", slider);
        if (slider) {
            settings[id] = parseInt(slider.value);
        }
    });

    let otherInputs = $(".dbInput");
    otherInputs.each(function () {
        let element = $(this);
        console.log("OTHER: ", element);
        let id = element.data("elem_id");
        settings[id] = element[0].value;
    });

    console.log("Collected settings: ", settings);
    return settings;
}


function dreamResponse() {

}

function startDreambooth() {

}
