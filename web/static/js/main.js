let previewNode = $("#template");
let previews = $("#previews");
let actions = $("#actions");
let controls = $(".controls");

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
        maxWidth: 200,
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
        placement: "top-start",
    })[0];
}

let drop = new Dropzone("div#actions", {
    url: "/edit",
    thumbnailWidth: 60,
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
    console.log("Added file");
    file.previewElement.querySelector(".start").onclick = function() { drop.enqueueFile(file); };

});

drop.on("sending", function(file, xhr, formData) {
    file.previewElement.querySelector(".start").setAttribute("disabled", "disabled");
    formData.append("filesize", file.size);
});

drop.on("dragenter", function() {
    actions.fadeTo( 0 , 0.5);
    // document.body
    infoTooltips[0].show();
    // actions.addClass('gray');
    $(".controls > button").prop("disabled", true);
});

drop.on("dragover", function(event) {
    event.preventDefault();
});

drop.on("dragleave", function() {
    actions.fadeTo( 0 , 1);
    infoTooltips[0].hide();
    $(".controls > button").prop("disabled", false);
    // actions.removeClass('gray');
});


drop.on("error", function(file, errorMessage) {
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
        content: `${res}<br/><span style="font-size: 11px;">Click anywhere to hide</span>`
    });
    statusTooltip.show();
});




document.querySelector("#actions .start").onclick = function() {
    drop.enqueueFiles(drop.getFilesWithStatus(Dropzone.ADDED));
};
