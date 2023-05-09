// Retrieve the value of checkboxValue and checkboxBottomValue from storage and set the default value to false if not found
chrome.storage.sync.get(
  { simplificationCheckboxValue: false, summarizationCheckboxValue: false },
  function (data) {
    const simplificationCheckbox = document.getElementById(
      "simplificationCheckbox"
    );
    const summarizationCheckbox = document.getElementById(
      "summarizationCheckbox"
    );
    simplificationCheckbox.checked = data.simplificationCheckboxValue;
    summarizationCheckbox.checked = data.summarizationCheckboxValue;
  }
);

document.addEventListener("DOMContentLoaded", function () {
  const simplificationCheckbox = document.getElementById(
    "simplificationCheckbox"
  );
  simplificationCheckbox.addEventListener("click", function () {
    const simplificationCheckboxValue = simplificationCheckbox.checked;
    chrome.storage.sync.set({
      simplificationCheckboxValue: simplificationCheckboxValue,
    });
    console.log(
      "simplificationCheckbox value set to:",
      simplificationCheckboxValue
    );

    // Reload the active tab to refresh actions.js
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.reload(tabs[0].id);
    });
  });

  const summarizationCheckbox = document.getElementById(
    "summarizationCheckbox"
  );
  summarizationCheckbox.addEventListener("click", function () {
    const summarizationCheckboxValue = summarizationCheckbox.checked;
    chrome.storage.sync.set({
      summarizationCheckboxValue: summarizationCheckboxValue,
    });
    console.log(
      "summarizationCheckbox bottom value set to:",
      summarizationCheckboxValue
    );

    // Reload the active tab to refresh actions.js
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.reload(tabs[0].id);
    });
  });
});
