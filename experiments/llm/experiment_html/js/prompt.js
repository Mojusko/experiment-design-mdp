// Script for prompt.html

const styleDescriptionInput = document.getElementById('style-description-input');
const saveFinalButton = document.getElementById('save-final-button');

saveFinalButton.addEventListener('click', () => {
    // Retrieve feedback data saved from index.html
    const feedbackData = JSON.parse(localStorage.getItem('imageFeedback')) || {};

    // Get the style description from this page
    const styleDescription = styleDescriptionInput.value.trim();

    // --- Validation ---
    // Check if style description is entered
    if (styleDescription === '') {
        alert("Please describe the style you had in mind before saving.");
        return; // Stop if description is missing
    }

    // Optional: Check if feedbackData is empty (user somehow skipped the feedback page)
    if (Object.keys(feedbackData).length === 0) {
        if (!confirm("Warning: No image preferences were found. This usually means the feedback steps were skipped.\n\nDo you want to save the prompts anyway?")) {
            return; // Stop if user cancels
        }
    }

    // --- Prepare data for JSON ---
    // Process feedbackData to include detailed info
    const formattedPreferences = [];
    const filenamePattern = /^images\/alg-([a-zA-Z0-9]+)_episode_(\d+)_timestep_(\d+)\.png$/i;

    for (const filename in feedbackData) {
        if (feedbackData.hasOwnProperty(filename)) {
            const match = filename.match(filenamePattern);
            if (match) {
                formattedPreferences.push({
                    filename: filename, // Keep original filename
                    algorithm: match[1],
                    episode: parseInt(match[2], 10),
                    timestep: parseInt(match[3], 10),
                    preference: feedbackData[filename] // The user's choice (1-based index)
                });
            } else {
                console.warn(`Could not parse filename in feedback data: ${filename}`);
                // Optionally include raw data if parsing fails
                // formattedPreferences.push({ filename: filename, preference: feedbackData[filename], error: "parse_failed" });
            }
        }
    }

    // Sort the preferences for consistency (optional, but good practice)
    formattedPreferences.sort((a, b) => {
        if (a.algorithm !== b.algorithm) return a.algorithm.localeCompare(b.algorithm);
        if (a.episode !== b.episode) return a.episode - b.episode;
        return a.timestep - b.timestep;
    });

    // Retrieve benchmark episode keys from localStorage
    const benchmarkEpisodeKeys = JSON.parse(localStorage.getItem('benchmarkEpisodeKeys')) || [];
    if (benchmarkEpisodeKeys.length > 0) {
        console.log("Retrieved benchmarkEpisodeKeys from localStorage:", benchmarkEpisodeKeys);
    }

    const outputData = {
        style_description: styleDescription, // Store the style description
        preferences: formattedPreferences, // Store the detailed, formatted preferences
        benchmark_episode_keys: benchmarkEpisodeKeys // Add benchmark keys
    };

    const outputJson = JSON.stringify(outputData, null, 2); // Pretty print JSON
    const blob = new Blob([outputJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    // Suggest a filename including the style description (sanitized)
    const sanitizedDescription = styleDescription.replace(/[^a-z0-9]/gi, '_').toLowerCase().substring(0, 30);
    a.download = `feedback_${sanitizedDescription || 'data'}.json`;
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
