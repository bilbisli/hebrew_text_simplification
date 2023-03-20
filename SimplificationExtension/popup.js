$(function () {
  $("#SimpText").click(function () {
    var search_topic = $("#textInput").val();

    if (search_topic) {
      chrome.runtime.sendMessage({ topic: search_topic }, function (response) {
        result = response.farewell;
        alert(result.summary);

        var notifOptions = {
          type: "basic",
          iconUrl: "icon48.png",
          title: "Simplification For Your Result",
          message: result.summary,
        };

        chrome.notifications.create("Simplification", notifOptions);
      });
    }

    $("#textInput").val("");
  });
});
