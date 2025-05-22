// Ensure imageFiles is loaded from imageList.js before this script runs
if (typeof imageFiles === 'undefined' || !Array.isArray(imageFiles) || imageFiles.length === 0) {
    alert("Error: imageFiles is not defined or empty. Make sure js/imageList.js is generated and loaded correctly.");
    // Disable functionality if images aren't loaded
    document.body.innerHTML = "<h1>Error loading image list. Please generate js/imageList.js</h1>";
    throw new Error("imageFiles not loaded."); // Stop script execution
}

// --- Global Variables ---
let currentQuestionIndex = 0; // Index for the shuffled question order
let displayOrder = []; // Holds the structured and shuffled image data
let feedbackData = JSON.parse(localStorage.getItem('imageFeedback')) || {}; // Load feedback

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
        const match = filename.match(/alg-([a-zA-Z0-9]+)_episode_(\d+)_timestep_(\d+)\.png$/i);
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

    // 2. Group by (algorithm, episode)
    const groupedByEpisode = structuredImages.reduce((acc, imgData) => {
        const key = `${imgData.algorithm}-${imgData.episode}`;
        if (!acc[key]) {
            acc[key] = [];
        }
        acc[key].push(imgData);
        return acc;
    }, {});

    // 3. Sort images within each group by timestep
    for (const key in groupedByEpisode) {
        groupedByEpisode[key].sort((a, b) => a.timestep - b.timestep);
    }

    // 4. Create a list of unique (algorithm, episode) keys and shuffle it
    const episodeKeys = Object.keys(groupedByEpisode);
    // Fisher-Yates (Knuth) Shuffle
    for (let i = episodeKeys.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [episodeKeys[i], episodeKeys[j]] = [episodeKeys[j], episodeKeys[i]];
    }
    console.log("Shuffled episode order:", episodeKeys);

    const NUM_BENCHMARK_EPISODES = 5; // Number of episodes to reserve for benchmark
    let benchmarkEpisodeKeys = [];
    let userEpisodeKeys = [...episodeKeys]; // Start with all episodes

    if (episodeKeys.length > NUM_BENCHMARK_EPISODES) {
        // Take the last NUM_BENCHMARK_EPISODES from the shuffled list as benchmarks
        benchmarkEpisodeKeys = episodeKeys.slice(-NUM_BENCHMARK_EPISODES);
        // The rest are for the user
        userEpisodeKeys = episodeKeys.slice(0, episodeKeys.length - NUM_BENCHMARK_EPISODES);
        
        console.log("Benchmark episodes (keys, not shown to user):", benchmarkEpisodeKeys);
        console.log("User-answerable episodes (keys):", userEpisodeKeys);
    } else {
        console.warn(`Not enough episodes (${episodeKeys.length}) to reserve ${NUM_BENCHMARK_EPISODES} for benchmark. All episodes will be user-answerable.`);
        // All episodes remain in userEpisodeKeys, benchmarkEpisodeKeys is empty
    }

    // 5. Create the final displayOrder based on ALL shuffled episode keys
    // All episodes will be shown to the user for feedback.
    // benchmarkEpisodeKeys are still identified and saved to localStorage
    // so that prompt.js can mark them in feedback.json for later benchmark evaluation.
    displayOrder = [];
    episodeKeys.forEach(key => { // Use the full episodeKeys list here
        displayOrder.push(...groupedByEpisode[key]); // Add all timesteps for this shuffled episode
    });

    if (benchmarkEpisodeKeys.length > 0) {
        // Log that these episodes are marked for benchmark, but will still be shown.
        console.log(`Identified ${benchmarkEpisodeKeys.length} episodes for benchmark evaluation (will still be shown to user).`);
        // Save benchmarkEpisodeKeys to localStorage for prompt.js
        try {
            localStorage.setItem('benchmarkEpisodeKeys', JSON.stringify(benchmarkEpisodeKeys));
            console.log("Saved benchmarkEpisodeKeys to localStorage.");
        } catch (e) {
            console.error("Error saving benchmarkEpisodeKeys to localStorage:", e);
        }
    } else {
        // Ensure localStorage is cleared or set to empty if no benchmark keys
        localStorage.setItem('benchmarkEpisodeKeys', JSON.stringify([]));
    }
    // This now reflects the total number of questions the user will answer.
    console.log(`Initialization complete. Total questions to be displayed to user (all episodes): ${displayOrder.length}`);

    // 6. Start the display
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
        localStorage.setItem('imageFeedback', JSON.stringify(feedbackData));
        console.log(`Saved feedback for ${filenameKey}: ${feedbackData[filenameKey]}`);
    } else {
        // Optional: Clear feedback if nothing is selected
        // delete feedbackData[filenameKey];
        // localStorage.setItem('imageFeedback', JSON.stringify(feedbackData));
        // If you want to explicitly save 'null' or 'undefined' when nothing is chosen:
        // delete feedbackData[currentImageFile];
        // localStorage.setItem('imageFeedback', JSON.stringify(feedbackData));
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
    window.location.href = 'prompt.html'; // Redirect to prompt page
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
