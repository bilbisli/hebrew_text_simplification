var serverhost = "http://127.0.0.1:8000";

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log('listener text: ' + request.text);
  var url =
    serverhost + "/get_simplified/?text=" + encodeURIComponent(request.text);

  console.log(url);

  fetch(url)
    .then((response) => response.json())
    .then((response) => sendResponse({ simplified_text_response: response }))
    .catch((error) => console.log(error));

  return true; // Will respond asynchronously.
});

// chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {      // backhand request handler
//   fetch(request.input, request.init).then(function(response) {
//       return response.text().then(function(text) {
//           sendResponse([{
//               body: text,
//               status: response.status,
//               statusText: response.statusText,
//           }, null]);
//       });
//   }, function(error) {
//       sendResponse([null, error]);
//   });
//   return true;
// });

