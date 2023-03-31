let dreamModelSelect;
let newDreamModelSelect;
let inputBrowser;
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

    inputBrowser = new FileBrowser(document.getElementById("dreamInputBrowser"),
    {
        "dropdown": true,
        "showInfo": false,
        "showTitle": false,
        "showSelectButton": true
        }
    );

    inferProgress = new ProgressGroup(document.getElementById("dreamProgress"), {
        "primary_status": "Status 1", // Status 1 text
        "secondary_status": "Status 2", // Status 2...
        "bar1_progress": 0, // Progressbar 1 position
        "bar2_progress": 0 // etc
    });

    // Gallery creation. Options can also be passed to .update()
    gallery = new InlineGallery(document.getElementById('dreamGallery'),
        {
            "thumbnail": true,
            "closeable": false,
            "show_maximize": true,
            "start_open": true
        }
    );

    $(".db-slider").BootstrapSlider();

    // Register the module with the UI. Icon is from boxicons by default.
    registerModule("Dreambooth", "moduleDreambooth", "cloud", false, 2);


    registerSocketMethod("train_dreambooth", "train_dreambooth", dreamResponse);


    keyListener.register("ctrl+Enter", "#dreamSettings", startDreambooth);
});


function dreamResponse() {

}

function startDreambooth() {

}
