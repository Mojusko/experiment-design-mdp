import random

# Load a large list of English words from a file
with open('/usr/share/dict/american-english', 'r') as f:
    all_words = [word.strip() for word in f if word.strip()]

# Randomly select 3000 unique words
selected_words = random.sample(all_words, 3000)

# Write them to 'vocabulary.txt'
with open('english.txt', 'w') as f:
    for word in selected_words:
        f.write(word + '\n')
