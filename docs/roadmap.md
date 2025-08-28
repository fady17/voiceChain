**
    *   **Phase 0: Component Sanity Check (Completed)**
    *   **Phase 1: The Linear "Dumb Pipe" (Current)**
    *   **Phase 2: Introducing Asynchrony and Streaming**
    *   **Phase 3: The Intelligent Orchestrator & State Management**
    *   **Phase 4: Advanced Metadata & Prosody Control**
    *   **Phase 5: Adaptation & Fine-Tuning**


## **Project Roadmap & Checklist**

### ✅ **Phase 0: Component Sanity Check**
*   [x] **LLM:** Validate `llama.cpp` with a GGUF model on M4 Metal.
*   [x] **STT:** Validate `mlx-whisper` with a local `large-v3` model.
*   [x] **TTS:** Validate `mlx-audio` with the `Kokoro-82M` model.
*   [x] **Methodology:** Establish the R&D Framework (Whitepaper, Roadmap, Journal, README).

### 🔲 **Phase 1: The Linear "Dumb Pipe" (Current Phase)**
*   [ ] **Audio I/O:** Implement basic microphone recording and speaker playback functionality.
*   [ ] **Pipeline Script (`pipeline_v1.py`):** Create the main script to orchestrate the synchronous flow.
*   [ ] **STT Integration:** Connect audio input to the `mlx-whisper` transcription function.
*   [ ] **LLM Integration:** Connect the STT text output to the `llama.cpp` inference function.
*   [ ] **TTS Integration:** Connect the LLM text output to the `mlx-audio` synthesis function.
*   [ ] **End-to-End Test:** Run the full pipeline, speak a sentence, and hear a response.
*   [ ] **Journal Entry:** Document the baseline performance (total latency) and key challenges of the synchronous pipeline.





### **Phase 3C: Barge-In Interruption and Echo Cancellation**

This is the final step to make the conversation feel truly natural.

**The Plan:**

1.  **Track Agent's Speech:** The `VoiceAgent` needs to know what it is currently saying. When the `tts_consumer` synthesizes a sentence, we will store that sentence in a new state variable, e.g., `self.currently_speaking_text`.

2.  **Implement Software Echo Cancellation (`is_echo`):** We will create a new method `is_echo(self, user_text: str) -> bool`. This method will compare the incoming `user_text` (from the STT) with `self.currently_speaking_text`.
    *   A simple, robust first version can use a normalized string similarity metric. For example, convert both strings to lowercase, remove punctuation, and check if one is a substring of the other or if their Levenshtein distance is very small.

3.  **Implement the Interruption Handler (`handle_barge_in`):** This is the core logic. When an utterance is detected during the `SPEAKING` state and our `is_echo` function returns `False`, we trigger the interruption.
    *   **Action 1: Silence the Agent.** Immediately call a new `self.player.interrupt()` method. This method needs to clear the player's queue and stop the current playback instantly.
    *   **Action 2: Cancel the Cognitive Pipeline.** The current `run_pipeline` task must be cancelled. We can get a handle to it (e.g., `self.processing_task`) and call `self.processing_task.cancel()`. This will stop the LLM and TTS from generating any more of the old response.
    *   **Action 3: Start the New Turn.** Immediately start a *new* `run_pipeline` task with the new, interrupting user text.


