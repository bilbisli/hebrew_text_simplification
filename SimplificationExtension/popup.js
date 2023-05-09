chrome.storage.sync.get(
  {
    simplificationCheckboxValue: false,
    summarizationCheckboxValue: false,
    fontSizeValue: "medium",
    textColorValue: "#ffffff",
  },
  function (data) {
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

    simplificationCheckbox.checked = data.simplificationCheckboxValue;
    summarizationCheckbox.checked = data.summarizationCheckboxValue;
    fontSizeSelection.value = data.fontSizeValue;

    textColor.value = data.textColorValue;
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

  const fontSizeSelection = document.getElementById("fontSizeSelect");

  var customFontSizeInput = document.getElementById("customFontSizeInput");

  customFontSizeInput.addEventListener("change", function () {
    fontSizeValue = customFontSizeInput.value;

    chrome.storage.sync.set({ fontSizeValue: fontSizeValue });
    console.log("fontSize value set to:", fontSizeValue);

    // Reload the active tab to refresh actions.js
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.reload(tabs[0].id);
    });
  });

  fontSizeSelection.addEventListener("change", function () {
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
    let textColorValue = textColor.value;
    chrome.storage.sync.set({ textColorValue: textColorValue });
    console.log("textColorValue value set to:", textColorValue);

    // Reload the active tab to refresh actions.js
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.reload(tabs[0].id);
    });
  });
});
