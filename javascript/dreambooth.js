function start_training_dreambooth(){
    requestProgress('db');
    gradioApp().querySelector('#db_error').innerHTML='';
    return args_to_array(arguments);
}

onUiUpdate(function(){
    check_progressbar('db', 'db_progressbar', 'db_progress_span', '', 'db_interrupt', 'db_preview', 'db_gallery')
})