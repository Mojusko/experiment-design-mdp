// Script for prompt.html

const styleDescriptionInput = document.getElementById('style-description-input');
const promptForm = document.getElementById('prompt-form');
const submitButton = document.getElementById('submit-button');
const formStatus = document.getElementById('form-status');
const lambdaInput = document.getElementById('lambda-input');
const feedbackJsonInput = document.getElementById('feedback-json-input');

const urlParams = new URLSearchParams(window.location.search);
const urlLambda = urlParams.get('lam') || urlParams.get('lambda');
const storedLambda = localStorage.getItem('currentLambda');
const selectedLambda = urlLambda || storedLambda || '0.1';
console.log(`Preparing prompt page for lambda=${selectedLambda}.`);

try {
    localStorage.setItem('currentLambda', selectedLambda);
} catch (err) {
    console.warn('Unable to store current lambda for prompt page:', err);
}

if (lambdaInput) {
    lambdaInput.value = selectedLambda;
}

function loadFeedbackFromStorage() {
    let feedbackData = {};
    const feedbackStorageKey = `imageFeedback_${selectedLambda}`;
    try {
        const stored = localStorage.getItem(feedbackStorageKey);
        if (stored) {
            feedbackData = JSON.parse(stored) || {};
        }
    } catch (err) {
        console.warn('Unable to parse stored feedback for prompt page:', err);
        feedbackData = {};
    }

    if (!Object.keys(feedbackData).length) {
        try {
            const legacyStored = localStorage.getItem('imageFeedback');
            if (legacyStored) {
                const legacyData = JSON.parse(legacyStored) || {};
                if (Object.keys(legacyData).length) {
                    feedbackData = legacyData;
                    localStorage.setItem(feedbackStorageKey, JSON.stringify(feedbackData));
                    console.log(`Migrated legacy feedback to ${feedbackStorageKey}.`);
                }
            }
        } catch (err) {
            console.warn('Failed to migrate legacy feedback data on prompt page:', err);
        }
    }

    return feedbackData;
}

function loadBenchmarkKeys() {
    const benchmarkStorageKey = `benchmarkEpisodeKeys_${selectedLambda}`;
    try {
        const stored = localStorage.getItem(benchmarkStorageKey);
        if (stored) {
            return JSON.parse(stored) || [];
        }
    } catch (err) {
        console.warn('Unable to parse stored benchmark keys for prompt page:', err);
    }

    try {
        const legacyStored = localStorage.getItem('benchmarkEpisodeKeys');
        if (legacyStored) {
            const legacyKeys = JSON.parse(legacyStored) || [];
            if (legacyKeys.length) {
                localStorage.setItem(benchmarkStorageKey, JSON.stringify(legacyKeys));
                console.log(`Migrated legacy benchmark keys to ${benchmarkStorageKey}.`);
                return legacyKeys;
            }
        }
    } catch (err) {
        console.warn('Failed to migrate legacy benchmark keys on prompt page:', err);
    }

    return [];
}

function buildOutputData(feedbackData, styleDescription) {
    const formattedPreferences = [];
    const filenamePattern = /alg-([a-zA-Z0-9]+)_episode_(\d+)_timestep_(\d+)\.png$/i;

    for (const filename in feedbackData) {
        if (Object.prototype.hasOwnProperty.call(feedbackData, filename)) {
            const baseName = filename.split('/').pop();
            const match = baseName.match(filenamePattern);
            if (match) {
                formattedPreferences.push({
                    filename,
                    algorithm: match[1],
                    episode: parseInt(match[2], 10),
                    timestep: parseInt(match[3], 10),
                    preference: feedbackData[filename]
                });
            } else {
                formattedPreferences.push({ filename, preference: feedbackData[filename] });
            }
        }
    }

    formattedPreferences.sort((a, b) => {
        if (a.algorithm !== b.algorithm) return String(a.algorithm).localeCompare(String(b.algorithm));
        if (a.episode !== b.episode) return (a.episode || 0) - (b.episode || 0);
        return (a.timestep || 0) - (b.timestep || 0);
    });

    const benchmarkEpisodeKeys = loadBenchmarkKeys();

    return {
        lambda: selectedLambda,
        style_description: styleDescription,
        preferences: formattedPreferences,
        benchmark_episode_keys: benchmarkEpisodeKeys,
        responses_recorded: formattedPreferences.length
    };
}

promptForm?.addEventListener('submit', async (event) => {
    event.preventDefault();

    if (!styleDescriptionInput) {
        return;
    }

    const styleDescription = styleDescriptionInput.value.trim();
    if (styleDescription === '') {
        alert('Please describe the style you had in mind before submitting.');
        return;
    }

    const feedbackData = loadFeedbackFromStorage();
    if (Object.keys(feedbackData).length === 0) {
        const proceed = confirm(
            'Warning: No image preferences were found. This usually means the feedback steps were skipped.\n\nDo you want to submit the prompts anyway?'
        );
        if (!proceed) {
            return;
        }
    }

    const outputData = buildOutputData(feedbackData, styleDescription);
    const feedbackJson = JSON.stringify(outputData, null, 2);

    if (feedbackJsonInput) {
        feedbackJsonInput.value = feedbackJson;
    }
    if (lambdaInput) {
        lambdaInput.value = selectedLambda;
    }

    const formData = new FormData(promptForm);
    formData.set('style_description', styleDescription);
    formData.set('lambda', selectedLambda);
    formData.set('feedback_json', feedbackJson);

    if (formStatus) {
        formStatus.style.display = 'block';
        formStatus.style.color = '#555';
        formStatus.textContent = 'Submitting your feedback...';
    }

    let submissionSucceeded = false;

    try {
        if (submitButton) {
            submitButton.disabled = true;
        }

        const response = await fetch(promptForm.action, {
            method: 'POST',
            body: formData,
            headers: { Accept: 'application/json' }
        });

        if (!response.ok) {
            throw new Error(`Submission failed with status ${response.status}`);
        }

        submissionSucceeded = true;
        if (formStatus) {
            formStatus.style.color = '#2a6';
            formStatus.textContent = 'Thank you! Your feedback was submitted successfully.';
        }
        if (submitButton) {
            submitButton.textContent = 'Submitted';
        }
        try {
            localStorage.removeItem(`imageFeedback_${selectedLambda}`);
            localStorage.removeItem(`benchmarkEpisodeKeys_${selectedLambda}`);
            localStorage.removeItem('imageFeedback_backup');
        } catch (cleanupErr) {
            console.warn('Unable to clear stored questionnaire data:', cleanupErr);
        }
    } catch (err) {
        console.error('Failed to submit feedback:', err);
        if (formStatus) {
            formStatus.style.color = '#c0392b';
            formStatus.textContent = 'Submission failed. Please check your connection and try again.';
        }
    } finally {
        if (!submissionSucceeded && submitButton) {
            submitButton.disabled = false;
        }
    }
});
