function tryModel(model_id) {
    window.location.href = "/try_model?model_id=" + model_id;
}

function visuals(model_id) {
    window.location.href = "/visuals?model_id=" + model_id;
}

function modelDetails(model_id) {
    window.location.href = "/model-history?model_id=" + model_id;
}

function report(model_id) {
    window.location.href = "/report?model_id=" + model_id;
}

function deploy(model_id) {
    window.location.href = "/deploy?model_id=" + model_id;
}

function metrics(model_id) {
    window.location.href = "/metrics?model_id=" + model_id;
}

function login(){
    window.location.href = "/login";
}

function signup(){
    window.location.href = "/signup"
}

function downloadSample(id) {
    window.location.href = "/sample-download?id=" + id
}

function home(){
    window.location.href = "/";
}


function hist(){
    window.location.href = "/history";
}

function samples(){
    window.location.href = "/sample-data";
}