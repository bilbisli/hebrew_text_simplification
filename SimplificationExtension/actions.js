var jQuery = window.$;

// using jQuery
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie != '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) == (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

var csrftoken = getCookie('csrftoken');


$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});

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
    return [null,null];
}

class PopupTooltip{
    constructor(range, template_html=null){
		this.range = range;
		this.contentHeight = 100;
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
		let strong_div = document.createElement("strong");
		strong_div.setAttribute("id", "tooltipp-title-b");
		strong_div.setAttribute("style", "12px");
		this.contentHeight += Math.ceil(title_text.length / 86.0) * 12;
		strong_div.textContent = title_text;
		title_html.appendChild(strong_div)
        return title_html.outerHTML;
    }

    createText(content_text, text_class='e'){
        let text_html = document.createElement("div");
        text_html.setAttribute("class", "sr_" + text_class);
		text_html.setAttribute("id", "tooltipp-text-content");
		text_html.setAttribute("style", "12px");
		this.contentHeight += Math.ceil(content_text.length / 86.0) * 12;
        text_html.textContent = content_text;
        return text_html.outerHTML;
    }

    addSection(snippet){

        this.popupHtml = snippet + this.popupHtml;
    }

    addTextSection(titleText, contentText){
        let titleHtml = this.createTitle(titleText);
        let contetTextHtml = this.createText(contentText);
        let textSection = titleHtml + contetTextHtml;
        this.addSection(textSection);
    }

    loadTemplate(template_html){
        // const template = document.createElement("tooltip");
        // var ajax = new XMLHttpRequest();

        // ajax.open("GET", template_html, false);
        // ajax.send();
        // template.innerHTML += ajax.responseText;
        // this.popupHtml = template.content;
		this.popupHtml = '';
    }

	reset(){
		this.popupHtml = '';
	}

	getPosition(){
		let x = this.range.left + this.range.width/2 + 200;
		let y = this.range.currentY - this.range.height;
		if(x > window.innerWidth)
		{
			x = window.innerWidth;
		}
		if(y - this.range.height < 400)
		{
			y = this.range.bottom + this.contentHeight + 90;
		}
		return {'x': x, 'y': y}
	}

    getHtml(xPos=null, yPos=null){
		if (xPos == null || yPos == null){
			let pos = this.getPosition();
			xPos = pos.x;
			yPos = pos.y;
		}
        let output = '<div class="tooltipp" style="top: ' + yPos + 'px !important; left: ' + (xPos) + 'px !important;">';
        output +=   '<span class="tooltipp-content">';
        output +=       '<span class="tooltipp-text">';
        output +=           this.popupHtml;
        output +=       '</span>';
        output +=   '</span>';
        output += '</div>';

        return output;
    }

}
let port = 8000;
let ip = 'http://127.0.0.1';
let microSeviceSubPath = 'get_simplified/?text=';

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
		console.log('click');
		var container = $(".tooltipp");

		if (!container.is(e.target) && container.has(e.target).length === 0)
		{
			$("#tooltip").remove();
			$(".tooltipp").remove();
		}
	});

	$("body").on("click", function(e) {
		console.log('click2');
        const selected = getSelected();
		console.log('selected: ' + selected);
		let TextElement = selected[2];
		console.log('TextElement: ' + TextElement);
		if(TextElement)
		{
			let selectedRange = selected[1];
			selectedRange.currentY = e.pageY;
			console.log(selectedRange);
            let st = selected[0];

			if (st != null && st.length != 0)
			{
	
				let toolTip = new PopupTooltip(selectedRange);
				chrome.runtime.sendMessage({text: st}, async function (response) {
					console.log('msg???');
					console.log('response: ' + response)
					simpleText = response.simplified_text_response.simple_text;
					console.log('ress???');
					console.log('result: ' + simpleText);
					toolTip.addTextSection('טקסט מפושט:', simpleText);
					console.log(toolTip.getHtml());
					ttHtml = toolTip.getHtml();
					$("body").append(ttHtml);
					toolTip.reset();
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

