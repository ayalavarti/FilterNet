let previewNode = $("#template");
let previews = $("#previews");
let info = $(".info");
let files = $(".files");

previewNode.id = "";

let previewTemplate = previewNode.parent().html();
previewNode.remove();

let infoTooltips;
let statusTooltip;

$(document).ready(function () {
    initTooltips();
    previews.show();
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

drop.on("addedfile", function(file) {
    file.previewElement.querySelector(".start").onclick = function() { drop.enqueueFile(file); };
    file.previewElement.querySelector(".edit").onclick = function() { drop.enqueueFile(file); };
});

drop.on("sending", function(file, xhr, formData) {
    file.previewElement.querySelector(".start").setAttribute("disabled", "disabled");
    formData.append("filesize", file.size);
    if (file.id !== undefined) {
        formData.append("id", file.id);
    }
});

drop.on("dragenter", function() {
    info.fadeTo( 0 , 0.5);
    files.fadeTo( 0 , 0.5);
    // document.body
    infoTooltips[0].show();
    // actions.addClass('gray');
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
    file.id = res["id"];
    statusTooltip.show();
    file.status = Dropzone.ADDED;
});

document.querySelector("#actions .start").onclick = function() {
    drop.enqueueFiles(drop.getFilesWithStatus(Dropzone.ADDED));
};
