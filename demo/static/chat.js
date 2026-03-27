let typingIndicator = null; // to keep track of the typing indicator element

async function send() {

    const input = document.getElementById("question")
    const question = input.value
    const modelSelect = document.getElementById("model-select"); // get the model selector
    const selectedModel = modelSelect.value; // get the selected model value

    if (!question) return

    addMessage(question, "user")
    
    input.value = ""
    input.disabled = true; // Disable input while waiting for response
    modelSelect.disabled = true; // Disable model selection while waiting for response
    
    // Add typing indicator
    typingIndicator = addMessage("...", "typing-indicator");


    const response = await fetch("/ask", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            question: question,
            model: selectedModel // include the selected model in the request
        })

    })

    const data = await response.json()

    // Remove typing indicator
    if (typingIndicator) {
        typingIndicator.remove();
        typingIndicator = null;
    }
    input.disabled = false; // Re-enable input
    modelSelect.disabled = false; // Re-enable model selection
    input.focus(); // Focus input for next message

    // check for error and display appropriate message
    if (data.answer.error) {
        // if message is an array (from unsupported query), join it
        let errorMessage = data.answer.message;
        if (Array.isArray(errorMessage)) {
            errorMessage = errorMessage.join("\n");
        }
        errorMessage = errorMessage.replace(/\*\*/g, ""); // Remove bold formatting
        addMessage(errorMessage, "bot");
    } else {
        // Pass the entire answer object for detailed formatting
        // Remove bold formatting from the explanation
        if (data.answer.explanation) {
            data.answer.explanation = data.answer.explanation.replace(/\*\*/g, "");
        }
        // Remove bold formatting from security recommendations
        if (data.answer.security_recommendations) {
            data.answer.security_recommendations = data.answer.security_recommendations.replace(/\*\*/g, "");
        }
        addMessage(data.answer, "bot");
    }
}


function addMessage(text, type) {

    const box = document.getElementById("chat-box")

    const msg = document.createElement("div")

    msg.className = type

    if (type === "typing-indicator") {
        msg.innerHTML = `<span></span><span></span><span></span>`;
    } else if (type === "bot" && typeof text === 'object' && text !== null) {
        // Format structured bot response
        let featureHtml = '';
        if (text.packet_features) {
            featureHtml += '<details><summary><strong>packet features</strong></summary><ul>';
            for (const key in text.packet_features) {
                featureHtml += `<li><strong>${key}:</strong> ${text.packet_features[key]}</li>`;
            }
            featureHtml += '</ul></details>';
        }

        let windowHtml = '';
        if (text.packet_window) {
            windowHtml += '<details><summary><strong>packet window details</strong></summary><ul>';
            windowHtml += `<li><strong>count:</strong> ${text.packet_window.packet_count}</li>`;
            
            // Display predicted events with timestamps
            if (text.packet_window.predicted_events && Object.keys(text.packet_window.predicted_events).length > 0) {
                windowHtml += '<li><details><summary><strong>predicted events</strong></summary><ul>';
                for (const label in text.packet_window.predicted_events) {
                    const timestamps = text.packet_window.predicted_events[label];
                    const formattedTimestamps = timestamps.map(ts => `<span class="timestamp-highlight">${ts}</span>`).join(', ');
                    windowHtml += `<li><strong>${label}</strong> (${timestamps.length} events): ${formattedTimestamps}</li>`;
                }
                windowHtml += '</ul></details></li>';
            } else {
                windowHtml += '<li><strong>predicted events:</strong> none</li>';
            }

            // Display actual events with timestamps
            if (text.packet_window.actual_events && Object.keys(text.packet_window.actual_events).length > 0) {
                windowHtml += '<li><details><summary><strong>actual events</strong></summary><ul>';
                for (const label in text.packet_window.actual_events) {
                    const timestamps = text.packet_window.actual_events[label];
                    const formattedTimestamps = timestamps.map(ts => `<span class="timestamp-highlight">${ts}</span>`).join(', ');
                    windowHtml += `<li><strong>${label}</strong> (${timestamps.length} events): ${formattedTimestamps}</li>`;
                }
                windowHtml += '</ul></details></li>';
            } else {
                windowHtml += '<li><strong>actual events:</strong> none</li>';
            }
            windowHtml += '</ul></details>';
        }

        let content = `
            <p class="event-detail"><strong>event at:</strong> <span class="timestamp-highlight">${text.timestamp}</span></p>
            <p class="event-detail"><strong>predicted (at <span class="timestamp-highlight">${text.timestamp}</span>):</strong> ${text.predicted}</p>
            <p class="event-detail"><strong>actual (at <span class="timestamp-highlight">${text.timestamp}</span>):</strong> ${text.actual}</p>
            <p class="event-detail"><strong>correct:</strong> ${text.correct ? 'yes' : 'no'}</p>
            ${featureHtml}
            ${windowHtml}
            <p>${text.explanation}</p>
        `;
        if (text.security_recommendations) {
            // For security recommendations, we can apply a general style to the whole recommendation text
            // since specific timestamps within it are generated by the LLM and not structured.
            // If the user wants to highlight timestamps within this raw text, the LLM needs to output
            // them with specific HTML tags, or a more complex client-side parsing would be needed.
            content += `<details class="security-recommendations-details"><summary><strong>security recommendations</strong></summary><p>${text.security_recommendations}</p></details>`;
        }
        msg.innerHTML = content;
    }
    else {
        msg.innerText = text
    }

    box.appendChild(msg)

    box.scrollTop = box.scrollHeight
    return msg; // Return the message element to allow removal for typing indicator
}

function clearChatBox() {
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML = ''; // Clear all child elements
}

async function loadSession(date) {
    clearChatBox(); // Clear current chat when loading a new session

    const response = await fetch(`/session/${date}`);
    const sessionData = await response.json();

    sessionData.forEach(msg => {
        addMessage(msg.question, "user");
        // when loading old sessions, also correctly display the bot's message
        if (msg.answer.error) {
            let errorMessage = msg.answer.message;
            if (Array.isArray(errorMessage)) {
                errorMessage = errorMessage.join("\n");
            }
            errorMessage = errorMessage.replace(/\*\*/g, ""); // Remove bold formatting
            addMessage(errorMessage, "bot");
        } else {
            // Pass the entire answer object for detailed formatting when loading sessions
            // Remove bold formatting from the explanation
            if (msg.answer.explanation) {
                msg.answer.explanation = msg.answer.explanation.replace(/\*\*/g, "");
            }
            // Remove bold formatting from security recommendations
            if (msg.answer.security_recommendations) {
                msg.answer.security_recommendations = msg.answer.security_recommendations.replace(/\*\*/g, "");
            }
            addMessage(msg.answer, "bot"); // Pass the full answer object
        }
    });
    
    // Update the header to show the current session date
    document.getElementById("session-date-header").innerText = `investigation session — ${date}`;
}

// Event listener for Enter key
document.addEventListener("DOMContentLoaded", () => {
    const questionInput = document.getElementById("question");
    questionInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            event.preventDefault(); // Prevent default Enter key behavior (e.g., new line)
            send();
        }
    });
});