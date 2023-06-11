chrome.storage.sync.get(
  {
    simplificationCheckboxValue: false,
    summarizationCheckboxValue: false,
    fontSizeValue: "medium",
    textColorValue: "#ffffff",
  },
  function (data) {
    // Retrieve values from storage and set them as initial values for the UI elements
    const simplificationCheckbox = document.getElementById(
      "simplificationCheckbox"
    );
    const summarizationCheckbox = document.getElementById(
      "summarizationCheckbox"
    );
    const fontSizeSelection = document.getElementById("fontSizeSelect");

    var fontSizeSelect = document.getElementById("fontSizeSelect");
    var customFontSizeInput = document.getElementById("customFontSizeInput");

    // show/hide custom font size input field when "custom" option is selected
    fontSizeSelect.addEventListener("change", function () {
      if (fontSizeSelect.value === "custom") {
        customFontSizeInput.style.display = "inline";
      } else {
        customFontSizeInput.style.display = "none";
      }
    });
    var textColor = document.getElementById("textColor");

    // Set the values of UI elements based on retrieved data from storage
    simplificationCheckbox.checked = data.simplificationCheckboxValue;
    summarizationCheckbox.checked = data.summarizationCheckboxValue;
    fontSizeSelection.value = data.fontSizeValue;
    textColor.value = data.textColorValue;
  }
);

document.addEventListener("DOMContentLoaded", function () {
  // Event listeners for UI elements
  const simplificationCheckbox = document.getElementById(
    "simplificationCheckbox"
  );
  simplificationCheckbox.addEventListener("click", function () {
    // Update storage value when simplification checkbox is clicked
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
    // Update storage value when summarization checkbox is clicked
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

  const fontSizeSelection = document.getElementById("fontSizeSelect");

  var customFontSizeInput = document.getElementById("customFontSizeInput");

  customFontSizeInput.addEventListener("change", function () {
    // Update storage value when custom font size input changes
    fontSizeValue = customFontSizeInput.value;
    chrome.storage.sync.set({ fontSizeValue: fontSizeValue });
    console.log("fontSize value set to:", fontSizeValue);

    // Reload the active tab to refresh actions.js
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.reload(tabs[0].id);
    });
  });

  fontSizeSelection.addEventListener("change", function () {
    // Update storage value when font size selection changes
    let fontSizeValue = fontSizeSelection.value;
    chrome.storage.sync.set({ fontSizeValue: fontSizeValue });
    console.log("fontSize value set to:", fontSizeValue);

    // Reload the active tab to refresh actions.js
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.reload(tabs[0].id);
    });
  });

  var textColor = document.getElementById("textColor");

  textColor.addEventListener("input", function () {
    // Update storage value when text color input changes
    let textColorValue = textColor.value;
    chrome.storage.sync.set({ textColorValue: textColorValue });
    console.log("textColorValue value set to:", textColorValue);

    // Reload the active tab to refresh actions.js
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.reload(tabs[0].id);
    });
  });
});
