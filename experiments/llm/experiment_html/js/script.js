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
const saveButton = document.getElementById('save-button');
const feedbackForm = document.getElementById('feedback-form');
const progressElement = document.getElementById('progress');

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

    // Load saved feedback for this image
    loadFeedback();

    // Update button states
    prevButton.disabled = currentIndex === 0;
    nextButton.disabled = currentIndex === imageFiles.length - 1;
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
        const radioToCheck = feedbackForm.elements['policy_preference'].querySelector(`[value="${savedValue}"]`);
        if (radioToCheck) {
            radioToCheck.checked = true;
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
    if (currentIndex < imageFiles.length - 1) {
        saveFeedback(); // Save feedback for the image we are leaving
        currentIndex++;
        updateImage();
    }
});

// Save feedback immediately when a radio button is clicked
feedbackForm.addEventListener('change', saveFeedback);

saveButton.addEventListener('click', () => {
    // Ensure latest selection is saved before exporting
    saveFeedback();

    // Count how many images have feedback
    const feedbackCount = Object.keys(feedbackData).length;
    const totalImages = imageFiles.length;
    const progressMessage = `Feedback collected for ${feedbackCount} out of ${totalImages} images.`;

    if (feedbackCount < totalImages) {
        if (!confirm(`${progressMessage}\n\nDo you want to save the incomplete feedback anyway?`)) {
            return; // User cancelled
        }
    } else {
         alert(`${progressMessage}\n\nSaving feedback...`);
    }


    const feedbackJson = JSON.stringify(feedbackData, null, 2); // Pretty print JSON
    const blob = new Blob([feedbackJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'feedback.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log("Feedback saved to feedback.json");
});

// --- Initial Load ---
updateImage();
