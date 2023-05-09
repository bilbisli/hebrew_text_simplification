let port = 8000;
let ip = "http://127.0.0.1";
let microSeviceSubPath = "get_simplified/?text=";

class Tooltip {
  constructor() {
    this.tooltipElement = null;
    this.selectedText = "";
    this.fontSize = "25px";
    this.textId = "tooltipp-text-content";
    this.contentId = "tooltipp-content";
    this.titleId = "tooltipp-title-b";
    this.textColor = "#ffffff";
  }

  createTitle(title_text) {
    let title_html = document.createElement("div");
    title_html.setAttribute("class", "sr_title");
    let strong_div = document.createElement("strong");
    strong_div.setAttribute("id", this.titleId);
    strong_div.setAttribute(
      "style",
      `font-size: ${this.fontSize} ; color: ${this.textColor};`
    );
    strong_div.textContent = title_text;
    title_html.appendChild(strong_div);
    return title_html;
  }

  createText(content_text, text_class = "e") {
    let text_html = document.createElement("div");
    text_html.setAttribute("class", "sr_" + text_class);
    text_html.setAttribute("id", this.textId);
    text_html.setAttribute(
      "style",
      `font-size: ${this.fontSize}; color: ${this.textColor};`
    );

    text_html.textContent = content_text;
    return text_html;
  }

  create() {
    this.tooltipElement = document.createElement("div");
    this.tooltipElement.classList.add("tooltipp");

    let tooltipContent = document.createElement("div");
    tooltipContent.classList.add(this.contentId);
    tooltipContent.setAttribute("id", this.contentId);
    let tooltipText = document.createElement("div");
    tooltipText.classList.add("tooltipp-text");

    let title = this.createTitle("טקסט מפושט");
    console.log(title.offsetHeight);
    tooltipText.appendChild(title);
    tooltipText.appendChild(this.createText(this.selectedText));
    tooltipContent.appendChild(tooltipText);

    tooltipContent.style.fontSize = `${this.fontSize}`;
    this.tooltipElement.appendChild(tooltipContent);

    this.tooltipElement.style.zIndex = "9999"; // Adjust the z-index value as needed
    this.tooltipElement.style.pointerEvents = "none"; // Allow events to pass through
    document.body.appendChild(this.tooltipElement);
  }

  show(selectedText, position, fontSizeValue, textColorValue) {
    this.selectedText = selectedText;
    this.textColor = textColorValue;
    let textElement = document.getElementById(this.textId);
    let contentElement = document.getElementById(this.contentId);
    let titleElement = document.getElementById(this.titleId);
    textElement.innerHTML = this.modifyText(this.selectedText);

    const range = window.getSelection().getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const scrollY = window.pageYOffset || document.documentElement.scrollTop;

    this.tooltipElement.style.left = `${rect.left}px`;
    this.tooltipElement.style.width = `${rect.width}px`; // Set width to match selected text
    this.tooltipElement.style.display = "block";

    if (!isNaN(fontSizeValue)) {
      fontSizeValue = `${fontSizeValue}px`;
    }
    this.fontSize = fontSizeValue;

    textElement.style.fontSize = fontSizeValue;
    titleElement.style.fontSize = fontSizeValue;

    textElement.style.color = textColorValue;
    titleElement.style.color = textColorValue;

    contentElement.style.fontSize = fontSizeValue;

    let tooltipHeight = contentElement.offsetHeight;
    console.log("tooltipHeight:" + tooltipHeight);
    let aboveSelection = rect.top + scrollY - tooltipHeight - 5;
    let belowSelection = rect.bottom + scrollY + 5;

    if (position === "bottom" && aboveSelection > 0) {
      this.tooltipElement.style.top = `${aboveSelection}px`;
    } else {
      this.tooltipElement.style.top = `${belowSelection}px`;
    }
  }
  hide() {
    this.tooltipElement.style.display = "none";
    this.selectedText = "";
  }

  modifyText(text) {
    // Modify the selected text here according to your requirements
    return text.toUpperCase();
  }
}

const tooltip = new Tooltip();
tooltip.create();
let simpleText = "";

document.addEventListener("mouseup", function (event) {
  const selectedText = window.getSelection().toString();
  chrome.storage.sync.get(
    [
      "simplificationCheckboxValue",
      "summarizationCheckboxValue",
      "fontSizeValue",
      "textColorValue",
    ],
    function (data) {
      const simplificationCheckboxValue = data.simplificationCheckboxValue;
      const summarizationCheckboxValue = data.summarizationCheckboxValue;
      const fontSizeValue = data.fontSizeValue;
      const textColorValue = data.textColorValue;

      console.log("textColorValue: " + textColorValue);
      if (selectedText.length > 0) {
        let TextElement = selectedText;

        if (TextElement) {
          if (
            event.target.id !== "simplificationCheckbox" &&
            event.target.id !== "summarizationCheckbox"
          ) {
            chrome.runtime.sendMessage(
              {
                text: TextElement,
                simplificationCheckbox: simplificationCheckboxValue,
                summarizationCheckbox: summarizationCheckboxValue,
              },
              async function (response) {
                simpleText = response.simplified_text_response.simple_text;

                const rect = window
                  .getSelection()
                  .getRangeAt(0)
                  .getBoundingClientRect();
                const position =
                  rect.bottom > window.innerHeight / 2 ? "bottom" : "top";

                tooltip.show(
                  simpleText,
                  position,
                  fontSizeValue,
                  textColorValue
                );
              }
            );
          }
        }
      }
    }
  );
});

document.addEventListener("mousedown", function (event) {
  tooltip.hide();
});

chrome.runtime.connect().onDisconnect.addListener(function () {
  // clean up when content script gets disconnected
  // console.log("cleaned");
});
