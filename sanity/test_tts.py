# test_tts.py (Fixed Version)

import time
import soundfile as sf
import numpy as np
from pathlib import Path

# CORRECT IMPORT: We import the specific function from the correct file/module.
from mlx_audio.tts.generate import generate_audio

# 1. Define constants
MODEL_NAME = "models/Kokoro"  # Keep as string for generate_audio
SPEAKER = "af_heart"
TEXT_TO_SYNTHESIZE = "This is a final, successful test of the Kokoro text-to-speech model, using the correct function signature."
OUTPUT_FILENAME = "kokoro_final_output"  # Remove .wav extension as it's added automatically

print("--- MLX Kokoro TTS Sanity Check (using generate_audio function) ---")

# --- Let's be more precise with our timing ---

# 2. Synthesize the audio using the high-level API
print(f"Loading model '{MODEL_NAME}' and synthesizing text with speaker '{SPEAKER}'...")
start_synth_time = time.time()

# The generate_audio function does not return the waveform directly.
# It's designed to save files or play audio. We will tell it to save a file.
# We set join_audio=False and stream=False to get a single output file.
generate_audio(
    text=TEXT_TO_SYNTHESIZE,
    model_path=MODEL_NAME,      # String for generate_audio
    voice=SPEAKER,              # Argument name from source: voice
    file_prefix=OUTPUT_FILENAME, # It constructs the full name inside
    audio_format="wav",         # Explicitly set the format
    join_audio=False,           # Ensure single file output
    stream=False,               # Ensure single file output
    play=False,                 # Do not play automatically
    verbose=True                # Get detailed performance metrics
)

synth_time = time.time() - start_synth_time
print(f"Total process (load + synthesis) completed in {synth_time:.2f} seconds.")

# 4. Verify the output and calculate metrics
print("\n--- Verifying Output and Calculating Metrics ---")
# The function may add a suffix like _000 to the filename
import glob
potential_files = glob.glob(f"{OUTPUT_FILENAME}*.wav")
if potential_files:
    output_file = potential_files[0]  # Use the first match
    print(f"Found output file: {output_file}")
else:
    output_file = f"{OUTPUT_FILENAME}.wav"  # Fallback to expected name

try:
    # The function saves the file as `file_prefix`.`audio_format`
    waveform, sample_rate = sf.read(output_file)
    print(f"✅ Successfully loaded output file: {output_file}")
    
    # Get file size for diagnostics
    import os
    file_size = os.path.getsize(output_file)
    print(f"   File size: {file_size:,} bytes")

    # The verbose output from generate_audio already gives us the RTF,
    # but we can calculate it ourselves for verification.
    audio_duration = len(waveform) / sample_rate
    # Note: synth_time here includes model loading again.
    # The verbose printout from the function will be more accurate.
    rtf = synth_time / audio_duration
    
    print("\n--- Architectural Metrics (Calculated) ---")
    print(f"Audio Duration: {audio_duration:.2f} seconds")
    print(f"Total Time (Load+Synth): {synth_time:.2f} seconds")
    print(f"Real-Time Factor (RTF): {rtf:.4f}")
    print("------------------------------------------")
    print("NOTE: The verbose output above shows detailed timing breakdown from the TTS engine.")
    
    # Performance assessment
    if rtf < 1.0:
        print(f"🚀 EXCELLENT: Running {1/rtf:.1f}x faster than real-time!")
    else:
        print(f"🐌 Running {rtf:.1f}x slower than real-time")

except FileNotFoundError:
    print(f"❌ ERROR: Output file '{output_file}' not found.")
    print("Available files:")
    import os
    for f in os.listdir('.'):
        if f.endswith('.wav'):
            print(f"  - {f}")
except Exception as e:
    print(f"❌ An error occurred while reading the output file: {e}")
    print(f"Attempted to read: {output_file}")
    print("Available files:")
    import os
    for f in os.listdir('.'):
        if f.endswith('.wav'):
            print(f"  - {f}")