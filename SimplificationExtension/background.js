var serverhost = "http://127.0.0.1:8000";

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  var url =
    serverhost + "/get_simplified/?text=" + encodeURIComponent(request.text);

  if (request.simplificationCheckbox !== null) {
    url += "&simplificationCheckbox=" + request.simplificationCheckbox;
  }

  if (request.summarizationCheckbox !== null) {
    url += "&summarizationCheckbox=" + request.summarizationCheckbox;
  }

  console.log(url);

  fetch(url)
    .then((response) => response.json())
    .then((response) => sendResponse({ simplified_text_response: response }))
    .catch((error) => console.log(error));

  return true; // Will respond asynchronously.
});
