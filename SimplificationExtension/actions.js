let port = 8000;
let ip = "http://127.0.0.1";
let microSeviceSubPath = "get_simplified/?text=";

class Tooltip {
  constructor() {
    this.tooltipElement = null;
    this.selectedText = "";
    this.fontSize = 25;
    this.textId = "tooltipp-text-content";
    this.contentId = "tooltipp-content";
  }

  createTitle(title_text) {
    let title_html = document.createElement("div");
    title_html.setAttribute("class", "sr_title");
    let strong_div = document.createElement("strong");
    strong_div.setAttribute("id", "tooltipp-title-b");
    strong_div.setAttribute("style", `font-size: ${this.fontSize}px`);
    strong_div.textContent = title_text;
    title_html.appendChild(strong_div);
    return title_html;
  }

  createText(content_text, text_class = "e") {
    let text_html = document.createElement("div");
    text_html.setAttribute("class", "sr_" + text_class);
    text_html.setAttribute("id", this.textId);
    text_html.setAttribute("style", `font-size: ${this.fontSize}px`);
    text_html.textContent = content_text;
    return text_html;
  }

  create() {
    this.tooltipElement = document.createElement("div");
    this.tooltipElement.classList.add("tooltipp");

    let tooltipContent = document.createElement("div");
    tooltipContent.classList.add(this.contentId);

    let tooltipText = document.createElement("div");
    tooltipText.classList.add("tooltipp-text");

    let title = this.createTitle("טקסט מפושט");
    console.log(title.offsetHeight);
    tooltipText.appendChild(title);
    tooltipText.appendChild(this.createText(this.selectedText));
    tooltipContent.appendChild(tooltipText);

    tooltipContent.style.fontSize = `${this.fontSize}px`;
    this.tooltipElement.appendChild(tooltipContent);

    this.tooltipElement.style.zIndex = "9999"; // Adjust the z-index value as needed
    this.tooltipElement.style.pointerEvents = "none"; // Allow events to pass through
    document.body.appendChild(this.tooltipElement);
  }

  show(selectedText, position) {
    this.selectedText = selectedText;
    let textElement = document.getElementById(this.textId);
    let contentElement = document.getElementById(this.contentId);
    textElement.innerHTML = this.modifyText(this.selectedText);

    const range = window.getSelection().getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const scrollY = window.pageYOffset || document.documentElement.scrollTop;

    this.tooltipElement.style.left = `${rect.left}px`;
    this.tooltipElement.style.width = `${rect.width}px`; // Set width to match selected text
    this.tooltipElement.style.display = "block";
    
    let tooltipHeight = textElement.offsetHeight + 55 + this.fontSize;
    let aboveSelection = rect.top + scrollY - tooltipHeight - 5;
    let belowSelection = rect.bottom + scrollY + 5;
    if (position === "bottom" && aboveSelection > 0) {
      console.log("above");
      this.tooltipElement.style.top = `${aboveSelection}px`;
    } else {
      console.log("below");
      console.log(aboveSelection);
      console.log(this.tooltipElement.offsetHeight);
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
  chrome.storage.sync.get("checkboxValue", function (data) {
    const checkboxValue = data.checkboxValue;
    if (selectedText.length > 0) {
      let TextElement = selectedText;
      console.log("TextElement: " + TextElement);
      if (TextElement) {
        if (event.target.id !== "checkbox") {
          chrome.runtime.sendMessage(
            { text: TextElement },
            async function (response) {
              console.log("msg???");
              console.log("response: " + response);
              if (checkboxValue == false) {
                simpleText = response.simplified_text_response.simple_text;
              } else {
                simpleText = response.simplified_text_response.summary;
              }
              console.log(checkboxValue);
              console.log("ress???");
              console.log("result: " + simpleText);

              const rect = window
                .getSelection()
                .getRangeAt(0)
                .getBoundingClientRect();
              const position =
                rect.bottom > window.innerHeight / 2 ? "bottom" : "top";

              tooltip.show(simpleText, position);
            }
          ); // pass checkboxValue as a parameter
        }
      }
    }
  });
});

document.addEventListener("mousedown", function (event) {
  tooltip.hide();
});

chrome.runtime.connect().onDisconnect.addListener(function () {
  // clean up when content script gets disconnected
  // console.log("cleaned");
});
