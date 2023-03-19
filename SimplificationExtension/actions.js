

var getSelected = function(){
    var t = '';
    if(window.getSelection) {
        t = window.getSelection();
    } else if(document.getSelection) {
        t = document.getSelection();
    } else if(document.selection) {
	t = document.selection.createRange().text;
    }
    if(t != '')
	{
		var getRange = t.getRangeAt(0);
		return [t.toString().trim().toLowerCase(), getRange.getBoundingClientRect(), t];  // 0: selected text, 1: location in document, 2: all text
	}
    return [null,null]
}

class PopupTooltip{
    constructor(template_html=null){
        if (template_html == null) {
            this.loadTemplate(template_html);
        }
        else {
            this.popupHtml = template_html;
        }
        
    }

    createTitle(title_text){
        let title_html = document.createElement("div");
        title_html.setAttribute("class", "sr_title");
        title_html.textContent = title_text;
        return title_html;
    }

    createText(content_text, text_class='e'){
        let text_html = document.createElement("div");
        text_html.setAttribute("class", "sr_" + text_class);
        text_html.textContent = content_text;
        return text_html;
    }

    addSection(snippet){
        this.popupHtml.prepend(snippet);
    }

    createTextSection(titleText, contentText){
        titleHtml = this.createTitle(titleText);
        contetTextHtml = this.createText(contentText);
        textSection = titleHtml + contetTextHtml;
        addSection(textSection);
    }

    loadTemplate(template_html){
        const template = document.createElement("tooltip");
        var ajax = new XMLHttpRequest();

        ajax.open("GET", template_html, false);
        ajax.send();
        template.innerHTML += ajax.responseText;
        this.popupHtml = template.content;
    }

    getHtml(){
        output = '<div class="tooltipp" style="top: ' + this.yPos + 'px !important; left: ' + (this.xPos) + 'px !important;">';
        output +=   '<span class="tooltipp-content">';
        output +=       '<span class="tooltipp-text">';
        output +=           '<b style="font-size: 12px;" id="tooltipp-text-b">' + this.popupHtml;
        output +=       '</span>';
        output +=   '</span>';
        output += '</div>';

        return output;
    }

}

$(document).ready(function(){
	//getWords();

/*	var input = "http://www.kizur.co.il/search_word.php?abbr=" + encodeURIComponent('מנכ"ל');
	console.log(input);
	chrome.runtime.sendMessage({input}, messageResponse => {
		const [response, error] = messageResponse;
		const parser = new DOMParser();
		var html = parser.parseFromString(response.body, "text/html");
		console.log(html.querySelector("td.sr_results:nth-child(2)").textContent);
	});*/

	
	$("body").on("click", function(e) {
		var container = $(".tooltipp")

		if (!container.is(e.target) && container.has(e.target).length === 0)
		{
			$("#tooltip").remove();
			$(".tooltipp").remove();
		}
	});

	$("body").on("click", function(e) {
        const selected = getSelected()
		let TextElement = selected[2];
		if(TextElement)
		{
			let range = selected[1];
            let st = selected[0]

			if (st != null && st.length != 0)
			{
				e.pageX = range.left + range.width/2 + 200;
				if(e.pageX > window.innerWidth)
				{
					e.pageX = window.innerWidth;
				}
				if(e.pageY - range.height < 400)
				{
					e.pageY += (range.height + 300);
				}
				else
				{
					e.pageY -= range.height;
				}

				let data = '{"text": "'+ st +'  "}';

				let init = {
					method: 'POST',
					body: data,
					headers: {
						'Content-Type': 'application/json',
					},
					allow_redirects: true
				}
                toolTip = PopupTooltip()
				chrome.runtime.sendMessage({input, init}, async function (messageResponse) {
					const [response, error] = messageResponse;
					let output = 
				});
			}
		}
	});
});

function stringToUint(string) {
	var string = btoa(unescape(encodeURIComponent(string))),
		charList = string.split(''),
		uintArray = [];
	for (var i = 0; i < charList.length; i++) {
		uintArray.push(charList[i].charCodeAt(0));
	}
	return new Uint8Array(uintArray);
}


function getWords() {

	var collectedText;

	$('p,h1,h2,h3,h4,h5').each(function(index, element){
		collectedText += element.innerText + "\n";
	});

	collectedText = collectedText.replace('undefined', '');

	collectedText = collectedText.replace(/[0-9]/g, '');

	console.log(collectedText);
}


//////////////////////////////


chrome.runtime.connect().onDisconnect.addListener(function() {
    // clean up when content script gets disconnected
	// console.log("cleaned");
});

