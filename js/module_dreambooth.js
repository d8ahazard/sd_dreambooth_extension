let dreamModelSelect;
let newDreamModelSelect;
$(".hide").hide();
document.addEventListener("DOMContentLoaded", function () {

    $(".modelSelect").modelSelect();
    $("#db_create_model").click(function(){
        let data = {};
        $(".newModelParam").each(function(index, elem) {
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
        sendMessage("create_dreambooth", data, false).then(()=>{
            console.log("New model params: ", data);
        });

    });

    $("#db_create_from_hub").change(function(){
       if ($(this).is(":checked")) {
           $("#hub_row").show();
           $("#local_row").hide();
       } else {
           $("#hub_row").hide();
           $("#local_row").show();
       }
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
