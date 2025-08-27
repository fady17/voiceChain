# test_stt.py (Final Confirmed Version - Local .safetensors Model)

import mlx_whisper
import time

# 1. Define constants
# This points to your local directory containing config.json and weights.safetensors
MODEL_PATH = "models/whisper-large-v3-turbo" # Use your exact directory name
AUDIO_FILE = "/Users/fady/Desktop/agents/voice/sanity/sanityTest.mp3"

print("--- MLX Whisper STT Sanity Check (using local .safetensors model) ---")

# 2. Transcribe the audio file
print(f"Loading model from '{MODEL_PATH}' and transcribing '{AUDIO_FILE}'...")
start_time = time.time()

result = mlx_whisper.transcribe(
    audio=AUDIO_FILE,
    path_or_hf_repo=MODEL_PATH,
    word_timestamps=True
)

transcription_time = time.time() - start_time
print("Transcription complete.")

# 3. Print the results
# (The rest of the script is identical to the previous version)
print("\n--- Transcription Result ---")
print(f"Transcription: {result['text'].strip()}")
print(f"Language: {result['language']}")
print("--------------------------")
print(f"Time taken for transcription: {transcription_time:.2f} seconds")

# 4. Examine word-level timestamps
print("\n--- Word-level Timestamps ---")
try:
    first_segment_words = result['segments'][0]['words']
    for word in first_segment_words[:15]:
        print(f"[{word['start']:.2f}s -> {word['end']:.2f}s] {word['word']}")
    if len(first_segment_words) > 15:
        print("...")
except (IndexError, KeyError) as e:
    print(f"Word-level timestamps not available or error processing them: {e}")
print("-----------------------------")