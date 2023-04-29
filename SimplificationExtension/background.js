var serverhost = "http://127.0.0.1:8000";

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log("listener text: " + request.text);
  var url =
    serverhost + "/get_simplified/?text=" + encodeURIComponent(request.text);

  if (request.checkbox) {
    console.log(request.checkbox);
  }
  console.log(url);

  fetch(url)
    .then((response) => response.json())
    .then((response) => sendResponse({ simplified_text_response: response }))
    .catch((error) => console.log(error));

  return true; // Will respond asynchronously.
});
