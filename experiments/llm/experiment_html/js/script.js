// Ensure imageFiles is loaded from imageList.js before this script runs
if (typeof imageFiles === 'undefined' || !Array.isArray(imageFiles) || imageFiles.length === 0) {
    alert("Error: imageFiles is not defined or empty. Make sure js/imageList.js is generated and loaded correctly.");
    // Disable functionality if images aren't loaded
    document.body.innerHTML = "<h1>Error loading image list. Please generate js/imageList.js</h1>";
    throw new Error("imageFiles not loaded."); // Stop script execution
}

let currentIndex = 0;
// Initialize feedback storage. Use localStorage for persistence across sessions.
let feedbackData = JSON.parse(localStorage.getItem('imageFeedback')) || {};

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

function updateImage() {
    const currentImageFile = imageFiles[currentIndex];
    imageElement.src = currentImageFile;
    imageElement.alt = `Image: ${currentImageFile}`;

    // Extract info from filename (adjust regex if format differs)
    const match = currentImageFile.match(/episode_(\d+)_timestep_(\d+)/);
    if (match) {
        imageInfoElement.textContent = `Episode ${parseInt(match[1], 10)}, Timestep ${parseInt(match[2], 10)}`;
    } else {
        imageInfoElement.textContent = currentImageFile.split('/').pop(); // Fallback to filename
    }

    // Update progress display
    progressElement.textContent = `${currentIndex + 1} / ${imageFiles.length}`;

    // --- Timestep 1 Handling ---
    const isTimestepOne = /timestep_01/.test(currentImageFile);
    feedbackForm.classList.toggle('disabled-feedback', isTimestepOne); // Add/remove class for styling

    policyRadioButtons.forEach((radio, index) => {
        radio.disabled = isTimestepOne;
        if (isTimestepOne && index === 0) {
            radio.checked = true; // Default check Policy 1
        }
    });

    if (isTimestepOne) {
        saveFeedback(); // Save the default feedback for timestep 1
    } else {
        // Load user's previous feedback only if not timestep 1
        loadFeedback();
    }
    // --- End Timestep 1 Handling ---


    // Update button states
    prevButton.disabled = currentIndex === 0;

    if (currentIndex === imageFiles.length - 1) {
        // Last image: Show Finish button, hide Next button
        nextButton.style.display = 'none';
        finishButton.style.display = 'inline-block'; // Or 'block' if preferred
        nextButton.disabled = true; // Keep it disabled logically
    } else {
        // Not the last image: Show Next button, hide Finish button
        nextButton.style.display = 'inline-block'; // Or 'block'
        finishButton.style.display = 'none';
        nextButton.disabled = false; // Enable next button
    }
}

function saveFeedback() {
    const currentImageFile = imageFiles[currentIndex];
    const selectedValue = feedbackForm.elements['policy_preference'].value;
    if (selectedValue) {
        // Store the selected value (as a number, 0 for skip)
        feedbackData[currentImageFile] = parseInt(selectedValue, 10);
        localStorage.setItem('imageFeedback', JSON.stringify(feedbackData)); // Save to localStorage
        console.log(`Saved feedback for ${currentImageFile}: ${feedbackData[currentImageFile]}`);
    } else {
        // If nothing is selected when moving away, ensure it's cleared or handled
        // Currently, we only save when a value *is* selected.
        // If you want to explicitly save 'null' or 'undefined' when nothing is chosen:
        // delete feedbackData[currentImageFile];
        // localStorage.setItem('imageFeedback', JSON.stringify(feedbackData));
    }
}

function loadFeedback() {
    const currentImageFile = imageFiles[currentIndex];
    const savedValue = feedbackData[currentImageFile];

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
    if (currentIndex > 0) {
        saveFeedback(); // Save feedback for the image we are leaving
        currentIndex--;
        updateImage();
    }
});

nextButton.addEventListener('click', () => {
    saveFeedback(); // Save feedback for the current image before moving
    if (currentIndex < imageFiles.length - 1) {
        // Move to the next image
        // Move to the next image
        currentIndex++;
        updateImage();
    }
    // Removed redirection logic - handled by finishButton now
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
                if (currentIndex < imageFiles.length - 1) {
                    console.log("Moving to next image...");
                    currentIndex++;
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
updateImage();
