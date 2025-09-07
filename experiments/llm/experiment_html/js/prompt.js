// Script for prompt.html

const styleDescriptionInput = document.getElementById('style-description-input');
const saveFinalButton = document.getElementById('save-final-button');
const backupButton = document.getElementById('backup-button');
const shareButton = document.getElementById('share-button');
const copyButton = document.getElementById('copy-button');
const saveStatus = document.getElementById('save-status');

function buildOutputData() {
    const feedbackData = JSON.parse(localStorage.getItem('imageFeedback')) || {};
    const styleDescription = styleDescriptionInput.value.trim();

    const formattedPreferences = [];
    const filenamePattern = /^images\/alg-([a-zA-Z0-9]+)_episode_(\d+)_timestep_(\d+)\.png$/i;

    for (const filename in feedbackData) {
        if (Object.prototype.hasOwnProperty.call(feedbackData, filename)) {
            const match = filename.match(filenamePattern);
            if (match) {
                formattedPreferences.push({
                    filename,
                    algorithm: match[1],
                    episode: parseInt(match[2], 10),
                    timestep: parseInt(match[3], 10),
                    preference: feedbackData[filename]
                });
            } else {
                // include raw if unexpected filename
                formattedPreferences.push({ filename, preference: feedbackData[filename] });
            }
        }
    }

    formattedPreferences.sort((a, b) => {
        if (a.algorithm !== b.algorithm) return String(a.algorithm).localeCompare(String(b.algorithm));
        if (a.episode !== b.episode) return (a.episode || 0) - (b.episode || 0);
        return (a.timestep || 0) - (b.timestep || 0);
    });

    const benchmarkEpisodeKeys = JSON.parse(localStorage.getItem('benchmarkEpisodeKeys')) || [];

    return {
        style_description: styleDescription,
        preferences: formattedPreferences,
        benchmark_episode_keys: benchmarkEpisodeKeys
    };
}

function saveLocalBackup(jsonStr) {
    try {
        const payload = { saved_at: new Date().toISOString(), data: JSON.parse(jsonStr) };
        localStorage.setItem('imageFeedback_backup', JSON.stringify(payload));
        if (saveStatus) {
            saveStatus.textContent = 'Saved local backup.';
            saveStatus.style.display = 'block';
        }
    } catch (e) {
        console.warn('Failed to save local backup:', e);
    }
}

async function shareResults(jsonStr, filename) {
    try {
        const file = new File([jsonStr], filename, { type: 'application/json' });
        if (navigator.canShare && navigator.canShare({ files: [file] })) {
            await navigator.share({
                title: 'Study feedback',
                text: 'Attached is my anonymized study feedback file.',
                files: [file]
            });
            return true;
        }
    } catch (err) {
        console.warn('Web Share failed:', err);
    }
    // Fallback: open mailto with instructions (attachment must be manual)
    const subject = encodeURIComponent('Study feedback JSON');
    const body = encodeURIComponent(
        'Hi,\n\nI have saved my anonymized study feedback file. I will attach it to this email.\n' +
        'If you cannot find the file, it is likely in your Downloads folder.\n\nThank you.'
    );
    const recipient = 'mmutny@broadinstitute.org';
    window.location.href = `mailto:${recipient}?subject=${subject}&body=${body}`;
    return false;
}

saveFinalButton.addEventListener('click', async () => {
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

    const outputData = buildOutputData();
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
    // Save a backup locally for reliability
    saveLocalBackup(outputJson);

    // Optional: Clear localStorage after saving
    // localStorage.removeItem('imageFeedback');
    // alert("Feedback saved successfully!");

    // Optional: Redirect or display a success message
    saveFinalButton.textContent = "Saved!";
    saveFinalButton.disabled = true;
    alert("Feedback saved successfully! Optionally share/email the results or save a local backup.");

});

// Backup button handler
backupButton?.addEventListener('click', () => {
    const outputData = buildOutputData();
    const outputJson = JSON.stringify(outputData, null, 2);
    saveLocalBackup(outputJson);
});

// Share/email handler
shareButton?.addEventListener('click', async () => {
    const description = styleDescriptionInput.value.trim();
    const sanitized = description.replace(/[^a-z0-9]/gi, '_').toLowerCase().substring(0, 30) || 'data';
    const filename = `feedback_${sanitized}.json`;
    const outputData = buildOutputData();
    const outputJson = JSON.stringify(outputData, null, 2);
    await shareResults(outputJson, filename);
});

// Copy JSON to clipboard handler
copyButton?.addEventListener('click', async () => {
    const outputData = buildOutputData();
    const outputJson = JSON.stringify(outputData, null, 2);
    try {
        await navigator.clipboard.writeText(outputJson);
        if (saveStatus) {
            saveStatus.textContent = 'Copied JSON to clipboard.';
            saveStatus.style.display = 'block';
        }
    } catch (e) {
        alert('Copy failed. Please select and copy manually.');
        console.warn('Clipboard copy failed:', e);
    }
});
