let previewNode = $("#template");
let previews = $("#previews");
let info = $(".info");
let files = $(".files");

let upload_disabled = false;

previewNode.id = "";

let previewTemplate = previewNode.parent().html();
previewNode.remove();

let infoTooltips;
let statusTooltip;
let startTooltip;

$(document).ready(function () {
    initTooltips();
    previews.show();
    $("#edited_image").hide();
    startTooltip.show();
});


/**
 * Initializes the page tooltips
 */
function initTooltips() {
    infoTooltips = tippy(".info", {
        animation: "scale",
        theme: "filternet",
        maxWidth: 280,
        arrow: true,
        arrowType: "round",
        inertia: true,
        sticky: true,
        placement: "bottom",
    });

    statusTooltip = tippy("#status", {
        animation: "scale",
        theme: "filternet",
        maxWidth: 250,
        trigger: 'manual',
        hideOnClick: true,
        inertia: true,
        arrow: false,
        sticky: true,
        allowHTML: true,
        placement: "top",
    })[0];

    startTooltip = tippy("#startTooltip", {
        animation: "scale",
        theme: "filternet-alt",
        trigger: 'manual',
        hideOnClick: false,
        inertia: true,
        arrow: true,
        sticky: true,
        allowHTML: true,
        placement: "bottom",
    })[0];
}

let drop = new Dropzone(document.body, {
    url: "/edit",
    thumbnailWidth: 60,
    maxThumbnailFilesize: 50,
    thumbnailHeight: 60,
    parallelUploads: 1,
    previewTemplate: previewTemplate,
    autoQueue: false,
    previewsContainer: "#previews",
    acceptedFiles: "image/*",
    clickable: ".fileinput-button",
    maxFiles: 4
});

function viewImage(file) {
    $("#edited_image").show();
    startTooltip.hide();
    console.log(file.image_url);
    $("#edited_image").attr("src", file.image_url);
}

drop.on("addedfile", function(file) {
    file.previewElement.querySelector(".start").onclick = function() { drop.enqueueFile(file); };
    file.previewElement.querySelector(".edit").onclick = function() { viewImage(file) };
    $("#upload-all").prop("disabled", false);
    upload_disabled = false;
});

drop.on("sending", function(file, xhr, formData) {
    let img = document.createElement('img');
    img.src = 'static/images/loading.gif';

    file.previewElement.querySelector(".start").setAttribute("disabled", "disabled");
    file.previewElement.querySelector(".loading").appendChild(img);
});

drop.on("dragenter", function() {
    info.fadeTo( 0 , 0.5);
    files.fadeTo( 0 , 0.5);
    infoTooltips[0].show();
    $(".controls > button").prop("disabled", true);
});

drop.on("dragover", function(event) {
    event.preventDefault();
});

$("body").click(function() {
    info.fadeTo( 0 , 1);
    files.fadeTo( 0 , 1);
    infoTooltips[0].hide();
    $(".controls > button").prop("disabled", false);
    if (upload_disabled) {
        $("#upload-all").prop("disabled", true);
    }
});

drop.on("drop", function() {
    info.fadeTo( 0 , 1);
    files.fadeTo( 0 , 1);
    infoTooltips[0].hide();
    $(".controls > button").prop("disabled", false);
});

drop.on("error", function(file, errorMessage) {
    console.log(errorMessage);
    drop.removeFile(file);
    statusTooltip.setProps({
        theme: "error",
        content: `${errorMessage}<br/><span style="font-size: 11px;">Click anywhere to hide</span>`
    });
    statusTooltip.show();
});

drop.on("success", function(file, res) {
    statusTooltip.setProps({
        theme: "success",
        content: `${res["status"]}<br/><span style="font-size: 11px;">Click anywhere to hide</span>`
    });
    statusTooltip.show();
    file.previewElement.querySelector(".loading").innerHTML= "";
    file.image_url = res["image_url"];
});

document.querySelector("#actions .start").onclick = function() {
    f = drop.getFilesWithStatus(Dropzone.ADDED);
    f.forEach(function (file) {
        file.previewElement.querySelector(".start").setAttribute("disabled", "disabled");
    });
    drop.enqueueFiles(f);

    upload_disabled = true;
    $("#upload-all").prop("disabled", true);
};
