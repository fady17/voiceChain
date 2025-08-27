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

