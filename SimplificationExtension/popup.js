$(function () {
  $("#SimpText").click(function () {
    var search_topic = $("#textInput").val();

    if (search_topic) {
      chrome.runtime.sendMessage({ topic: search_topic }, function (response) {
        result = response.farewell;

        var html =
          "<div id='simplified-text' dir='rtl'>" +
          "<h3>Simplified Text</h3>" +
          "<p class='result'>" +
          result.summary +
          "</p>" +
          "</div>";

        var popupWindow = window.open(
          "",
          "Simplified Text",
          "width=400,height=300"
        );
        popupWindow.document.write(html);

        var style = popupWindow.document.createElement("style");
        style.innerHTML =
          "#simplified-text { padding: 20px; background-color: #ffffff; border: 1px solid #cccccc; border-radius: 5px; box-shadow: 0 0 5px #cccccc; text-align: center; font-size: 1.2em; line-height: 1.5em; } h3 { margin: 0; font-size: 1.5em; font-weight: bold; } .result { margin-top: 10px; padding: 10px 0; } button { display: block; margin: 20px auto 0; background-color: #5cb85c; border: none; border-radius: 5px; color: #ffffff; font-size: 1.2em; font-weight: bold; padding: 10px 20px; cursor: pointer; } button:hover { background-color: #4cae4c; }";
        popupWindow.document.head.appendChild(style);

        var closeButton = popupWindow.document.createElement("button");
        closeButton.innerHTML = "Close";
        closeButton.addEventListener("click", function () {
          popupWindow.close();
        });

        popupWindow.document.body.appendChild(closeButton);
      });
    }

    $("#textInput").val("");
  });
});
