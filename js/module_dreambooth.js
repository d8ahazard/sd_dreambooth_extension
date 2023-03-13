let dreamModelSelect;

document.addEventListener("DOMContentLoaded", function () {

    dreamModelSelect = new ModelSelect(document.getElementById("dreamModelSelect"), {
        label: "Model Selection:",
        load_on_select: true, // Enable auto-loading model on selected...needs code in modelHandler
        "model_type": "dreambooth"
    });
    // Register the module with the UI. Icon is from boxicons by default.
    registerModule("Dreambooth", "moduleDreambooth", "cloud", false);


    registerSocketMethod("train_dreambooth", "train_dreambooth", dreamResponse);


    keyListener.register("ctrl+Enter", "#dreamSettings", startDreambooth);
});


function dreamResponse() {

}

function startDreambooth() {

}
