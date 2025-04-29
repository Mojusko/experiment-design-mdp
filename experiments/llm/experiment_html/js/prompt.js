// Script for prompt.html

const userPromptInput = document.getElementById('user-prompt-input');
const saveFinalButton = document.getElementById('save-final-button');

saveFinalButton.addEventListener('click', () => {
    // Retrieve feedback data saved from index.html
    const feedbackData = JSON.parse(localStorage.getItem('imageFeedback')) || {};

    // Get the user prompt from this page
    const userPrompt = userPromptInput.value.trim();

    // --- Validation ---
    // Check if prompt is entered
    if (!userPrompt) {
        alert("Please enter a content prompt before saving.");
        return; // Stop if prompt is missing
    }

    // Optional: Check if feedbackData is empty (user somehow skipped the feedback page)
    if (Object.keys(feedbackData).length === 0) {
        if (!confirm("Warning: No image preferences were found. This usually means the feedback steps were skipped.\n\nDo you want to save just the prompt anyway?")) {
            return; // Stop if user cancels
        }
    }

    // --- Prepare data for JSON ---
    const outputData = {
        user_prompt: userPrompt,
        preferences: feedbackData // Store image preferences under 'preferences' key
    };

    const outputJson = JSON.stringify(outputData, null, 2); // Pretty print JSON
    const blob = new Blob([outputJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    // Suggest a filename including the prompt (sanitized)
    const sanitizedPrompt = userPrompt.replace(/[^a-z0-9]/gi, '_').toLowerCase().substring(0, 30);
    a.download = `feedback_${sanitizedPrompt || 'data'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log("Feedback saved to JSON file.");

    // Optional: Clear localStorage after saving
    // localStorage.removeItem('imageFeedback');
    // alert("Feedback saved successfully!");

    // Optional: Redirect or display a success message
    saveFinalButton.textContent = "Saved!";
    saveFinalButton.disabled = true;
    alert("Feedback saved successfully! You can close this page.");

});
