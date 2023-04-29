// Retrieve the value of checkboxValue from storage and set the default value to false if not found
chrome.storage.sync.get({ checkboxValue: false }, function (data) {
  const checkbox = document.getElementById("checkbox");
  checkbox.checked = data.checkboxValue;
});

document.addEventListener("DOMContentLoaded", function () {
  const checkbox = document.getElementById("checkbox");
  checkbox.addEventListener("click", function () {
    const checkboxValue = checkbox.checked;
    chrome.storage.sync.set({ checkboxValue: checkboxValue });
    console.log("Checkbox value set to:", checkboxValue);

    // Reload the active tab to refresh actions.js
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.reload(tabs[0].id);
    });
  });
});
