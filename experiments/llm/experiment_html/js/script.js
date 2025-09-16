// Ensure imageFiles is loaded from imageList.js before this script runs
if (typeof imageFiles === 'undefined' || !Array.isArray(imageFiles) || imageFiles.length === 0) {
    alert("Error: imageFiles is not defined or empty. Make sure js/imageList.js is generated and loaded correctly.");
    // Disable functionality if images aren't loaded
    document.body.innerHTML = "<h1>Error loading image list. Please generate js/imageList.js</h1>";
    throw new Error("imageFiles not loaded."); // Stop script execution
}

const selectedLambda = window.selectedLambda || window.defaultLambdaKey || '0.1';
const feedbackStorageKey = `imageFeedback_${selectedLambda}`;
const benchmarkStorageKey = `benchmarkEpisodeKeys_${selectedLambda}`;
console.log(`Loading questionnaire for lambda=${selectedLambda}.`);

// --- Global Variables ---
let currentQuestionIndex = 0; // Index for the shuffled question order
let displayOrder = []; // Holds the structured and shuffled image data
let feedbackData = {};

try {
    const stored = localStorage.getItem(feedbackStorageKey);
    if (stored) {
        feedbackData = JSON.parse(stored) || {};
    }
} catch (err) {
    console.warn('Unable to parse stored feedback for key', feedbackStorageKey, err);
    feedbackData = {};
}

if (!Object.keys(feedbackData).length) {
    // Backwards compatibility with legacy key when switching lambdas
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
        console.warn('Unable to migrate legacy feedback data:', err);
    }
}

try {
    localStorage.setItem('currentLambda', selectedLambda);
} catch (err) {
    console.warn('Unable to store current lambda selection:', err);
}

// --- DOM Elements ---
const imageElement = document.getElementById('current-image');
const imageInfoElement = document.getElementById('image-info');
const prevButton = document.getElementById('prev-button');
const nextButton = document.getElementById('next-button');
const finishButton = document.getElementById('finish-button'); // Added finish button reference
// Removed saveButton reference
const feedbackForm = document.getElementById('feedback-form');
const progressElement = document.getElementById('progress');
// Removed userPromptInput reference
// Determine the number of policies dynamically by counting radio buttons
const policyRadioButtons = feedbackForm.elements['policy_preference'];
const numPolicies = policyRadioButtons.length;
console.log(`Detected ${numPolicies} policies.`);

// --- Initialization ---
function initializeQuestionnaire() {
    // 1. Parse imageFiles into structured data
    const structuredImages = imageFiles.map(filename => {
        const baseName = filename.split('/').pop();
        const match = baseName.match(/alg-([a-zA-Z0-9]+)_episode_(\d+)_timestep_(\d+)\.png$/i);
        if (match) {
            return {
                filename: filename,
                algorithm: match[1],
                episode: parseInt(match[2], 10),
                timestep: parseInt(match[3], 10)
            };
        }
        console.warn(`Could not parse filename: ${filename}`);
        return null; // Handle potential parsing errors
    }).filter(item => item !== null); // Remove null entries

    if (structuredImages.length === 0) {
        alert("Error: No valid image filenames found in imageList.js. Cannot proceed.");
        throw new Error("No valid images parsed.");
    }

    // 2. Group images by algorithm, then by episode
    const groupedByAlgorithm = structuredImages.reduce((acc, img) => {
        if (!acc[img.algorithm]) {
            acc[img.algorithm] = {};
        }
        const episodeKey = `ep_${img.episode}`;
        if (!acc[img.algorithm][episodeKey]) {
            acc[img.algorithm][episodeKey] = [];
        }
        acc[img.algorithm][episodeKey].push(img);
        return acc;
    }, {});

    // 3. Sort images within each episode by timestep
    for (const alg in groupedByAlgorithm) {
        for (const epKey in groupedByAlgorithm[alg]) {
            groupedByAlgorithm[alg][epKey].sort((a, b) => a.timestep - b.timestep);
        }
    }

    // 4. Select a balanced set of benchmark episodes
    const NUM_BENCHMARK_PER_ALG = 10;
    let benchmarkEpisodeKeys = [];
    let allEpisodeKeys = [];

    for (const alg in groupedByAlgorithm) {
        let episodeNumbers = Object.keys(groupedByAlgorithm[alg]).map(epKey => parseInt(epKey.replace('ep_', ''), 10));
        
        // Fisher-Yates Shuffle for this algorithm's episodes
        for (let i = episodeNumbers.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [episodeNumbers[i], episodeNumbers[j]] = [episodeNumbers[j], episodeNumbers[i]];
        }

        if (episodeNumbers.length > NUM_BENCHMARK_PER_ALG) {
            const benchmarkEpisodesForAlg = episodeNumbers.slice(-NUM_BENCHMARK_PER_ALG);
            benchmarkEpisodesForAlg.forEach(epNum => benchmarkEpisodeKeys.push(`${alg}-${epNum}`));
        } else {
            console.warn(`Algorithm '${alg}' has only ${episodeNumbers.length} episodes. Cannot reserve ${NUM_BENCHMARK_PER_ALG} for benchmark. Using all as non-benchmark.`);
        }
        
        // Add all episode keys for this algorithm to the master list for display shuffling
        episodeNumbers.forEach(epNum => allEpisodeKeys.push(`${alg}-${epNum}`));
    }

    console.log("Selected Benchmark Episodes (internal use, not shown to user):", benchmarkEpisodeKeys);
    localStorage.setItem(benchmarkStorageKey, JSON.stringify(benchmarkEpisodeKeys));

    // 5. Shuffle the combined list of all episode keys to randomize the user's viewing order
    for (let i = allEpisodeKeys.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [allEpisodeKeys[i], allEpisodeKeys[j]] = [allEpisodeKeys[j], allEpisodeKeys[i]];
    }
    console.log("Final shuffled order of all episodes for user:", allEpisodeKeys);

    // 6. Create the final displayOrder based on the fully shuffled episode keys
    displayOrder = [];
    allEpisodeKeys.forEach(key => {
        const [alg, epNum] = key.split('-');
        const epKey = `ep_${epNum}`;
        if (groupedByAlgorithm[alg] && groupedByAlgorithm[alg][epKey]) {
            displayOrder.push(...groupedByAlgorithm[alg][epKey]);
        }
    });

    console.log(`Initialization complete. Total questions to be displayed: ${displayOrder.length}`);

    // 7. Start the display
    updateImage();
}


function updateImage() {
    if (displayOrder.length === 0 || currentQuestionIndex >= displayOrder.length) {
        console.error("Error: displayOrder is empty or index out of bounds.");
        imageInfoElement.textContent = "Error loading questions.";
        return;
    }

    const currentImageData = displayOrder[currentQuestionIndex];
    imageElement.src = currentImageData.filename; // Use original filename for src
    imageElement.alt = `Question ${currentQuestionIndex + 1}`; // Alt text

    // Update title and progress
    imageInfoElement.textContent = `Question ${currentQuestionIndex + 1} / ${displayOrder.length}`;
    progressElement.textContent = `${currentQuestionIndex + 1} / ${displayOrder.length}`;

    // Load user's previous feedback
    loadFeedback();

    // Update button states
    prevButton.disabled = currentQuestionIndex === 0;

    if (currentQuestionIndex === displayOrder.length - 1) {
        // Last image: Show Finish button, hide Next button
        nextButton.style.display = 'none';
        finishButton.style.display = 'inline-block';
        nextButton.disabled = true;
    } else {
        // Not the last image: Show Next button, hide Finish button
        nextButton.style.display = 'inline-block';
        finishButton.style.display = 'none';
        nextButton.disabled = false;
    }
}

function saveFeedback() {
    if (currentQuestionIndex >= displayOrder.length) return; // Safety check

    const currentImageData = displayOrder[currentQuestionIndex];
    const filenameKey = currentImageData.filename; // Use the original filename as the key
    const selectedValue = feedbackForm.elements['policy_preference'].value;

    if (selectedValue) {
        feedbackData[filenameKey] = parseInt(selectedValue, 10);
        localStorage.setItem(feedbackStorageKey, JSON.stringify(feedbackData));
        console.log(`Saved feedback for ${filenameKey}: ${feedbackData[filenameKey]}`);
    } else {
        // Optional: Clear feedback if nothing is selected
        // delete feedbackData[filenameKey];
        // localStorage.setItem(feedbackStorageKey, JSON.stringify(feedbackData));
        // If you want to explicitly save 'null' or 'undefined' when nothing is chosen:
        // delete feedbackData[currentImageFile];
        // localStorage.setItem(feedbackStorageKey, JSON.stringify(feedbackData));
    }
}

function loadFeedback() {
    if (currentQuestionIndex >= displayOrder.length) return; // Safety check

    const currentImageData = displayOrder[currentQuestionIndex];
    const filenameKey = currentImageData.filename; // Use the original filename as the key
    const savedValue = feedbackData[filenameKey];

    // Reset all radio buttons first
    feedbackForm.reset(); // Clears selection

    if (savedValue !== undefined && savedValue !== null) {
        // Iterate through the radio buttons to find the one with the matching value
        for (const radio of policyRadioButtons) {
            if (radio.value === String(savedValue)) { // Compare value as string
                radio.checked = true;
                break; // Found the button, exit loop
            }
        }
    }
}

// --- Event Listeners ---

prevButton.addEventListener('click', () => {
    if (currentQuestionIndex > 0) {
        saveFeedback(); // Save feedback for the image we are leaving
        currentQuestionIndex--;
        updateImage();
    }
});

nextButton.addEventListener('click', () => {
    saveFeedback(); // Save feedback for the current image before moving
    if (currentQuestionIndex < displayOrder.length - 1) {
        currentQuestionIndex++;
        updateImage();
    }
    // Finish button handles the last image case
});

// --- Finish Button Listener ---
finishButton.addEventListener('click', () => {
    saveFeedback(); // Save feedback for the last image
    console.log("Finish button clicked. Redirecting to prompt page.");
    const queryString = window.location.search;
    window.location.href = queryString ? `prompt.html${queryString}` : 'prompt.html';
});

// Save feedback immediately when a radio button is clicked
feedbackForm.addEventListener('change', saveFeedback);

// Removed saveButton event listener

// --- Keyboard Shortcut Listener ---
document.addEventListener('keydown', (event) => {
    // Ignore if modifier keys are pressed (e.g., Ctrl+1)
    if (event.ctrlKey || event.altKey || event.metaKey || event.shiftKey) {
        return;
    }

    const key = event.key;
    // Check if the key is a digit from 1 to numPolicies
    if (/^[1-9]$/.test(key)) {
        const selectedPolicy = parseInt(key, 10);

        if (selectedPolicy >= 1 && selectedPolicy <= numPolicies) {
            // Find the corresponding radio button by iterating
            let radioToCheck = null;
            console.log(`Searching for radio button with value: "${String(selectedPolicy)}"`); // Log target value
            for (const radio of policyRadioButtons) {
                console.log(`  Checking radio button value: "${radio.value}" (type: ${typeof radio.value})`); // Log current radio value and type
                if (radio.value === String(selectedPolicy)) {
                    console.log(`  Match found!`); // Log match
                    radioToCheck = radio;
                    break;
                }
            }

            if (radioToCheck) {
                console.log(`Key ${selectedPolicy} pressed, selecting Policy ${selectedPolicy}`);
                // Select the radio button
                radioToCheck.checked = true;
                // Call saveFeedback directly after checking the button
                saveFeedback();

                // Move to the next image if not the last one
                if (currentQuestionIndex < displayOrder.length - 1) {
                    console.log("Moving to next question...");
                    currentQuestionIndex++;
                    updateImage();
                } else {
                    // If it was the last image, simulate finish button click
                    console.log("Key pressed on last image. Finishing...");
                    finishButton.click(); // Trigger the finish button's action
                }
                // Prevent default browser action for the number key (e.g., scrolling)
                event.preventDefault();
            } else {
                // Log if no matching radio button was found after the loop
                console.log(`  No radio button found with value "${String(selectedPolicy)}".`);
            }
        }
    } else if (key === 'ArrowLeft') {
        // Simulate click on Previous button if enabled
        if (!prevButton.disabled) {
            console.log("Left arrow pressed, going previous...");
            prevButton.click();
            event.preventDefault(); // Prevent default browser action (scrolling)
        }
    } else if (key === 'ArrowRight') {
        // Check if Finish button is visible (last image)
        if (finishButton.style.display !== 'none') {
            console.log("Right arrow pressed on last image, finishing...");
            finishButton.click();
        } else if (!nextButton.disabled) {
            // Otherwise, simulate click on Next button if enabled
            console.log("Right arrow pressed, going next...");
            nextButton.click();
        }
        event.preventDefault(); // Prevent default browser action (scrolling)
    }
});


// --- Initial Load ---
initializeQuestionnaire(); // Parse, shuffle, and load the first image
