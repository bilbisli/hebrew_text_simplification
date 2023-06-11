// Define the server host URL
var serverhost = "http://127.0.0.1:8000";

// Event listener for incoming messages from the extension's content script
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  // Construct the URL for the API endpoint
  var url =
    serverhost + "/get_simplified/?text=" + encodeURIComponent(request.text);

  // Append query parameters based on request options

  // Append simplification checkbox value if not null
  if (request.simplificationCheckbox !== null) {
    url += "&simplificationCheckbox=" + request.simplificationCheckbox;
  }

  // Append summarization checkbox value if not null
  if (request.summarizationCheckbox !== null) {
    url += "&summarizationCheckbox=" + request.summarizationCheckbox;
  }

  console.log(url);

  // Make a fetch request to the constructed URL
  fetch(url)
    .then((response) => response.json()) // Parse the response as JSON
    .then((response) => sendResponse({ simplified_text_response: response })) // Send the response back to the content script
    .catch((error) => console.log(error)); // Log any errors that occur

  return true; // Will respond asynchronously.
});