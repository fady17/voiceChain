## **Project Roadmap & Checklist**

### ✅ **Phase 0: Component Sanity Check**
*   [x] **LLM:** Validate `llama.cpp` with a GGUF model on M4 Metal.
*   [x] **STT:** Validate `mlx-whisper` with a local `large-v3` model.
*   [x] **TTS:** Validate `mlx-audio` with the `Kokoro-82M` model.
*   [x] **Methodology:** Establish the R&D Framework (Whitepaper, Roadmap, Journal, README).

### ✅ **Phase 1: The Linear "Dumb Pipe" (Completed)**
*   [x] All tasks complete.
*   [x] **Outcome:** Established a baseline "dead air" latency of **~6 seconds**. Proved the synchronous model is not viable for real-time conversation.

### 🔲 **Phase 2: Introducing Asynchrony and Streaming (Current Phase)**
*   [ ] **Task 2.1: Refactor to a Persistent Server Architecture.** Convert `pipeline_v1.py` into a long-running application where models are loaded only once at startup.
*   [ ] **Task 2.2: Implement LLM Token Streaming.** Modify the `LLMEngine` to `yield` tokens as they are generated, rather than returning the full response at the end.
*   [ ] **Task 2.3: Implement "First-Chunk" TTS.** Modify the `TextToSpeechEngine` to accept a stream of text. It will buffer text until it forms a complete sentence, synthesize that chunk of audio, and immediately send it for playback.
*   [ ] **Task 2.4: The Asynchronous Orchestrator.** Replace the linear `if __name__ == "__main__"` block with an `asyncio` event loop. Use `asyncio.Queue` to create non-blocking pipes between the STT, LLM, and TTS components.
*   [ ] **Task 2.5: Implement Streaming Audio I/O.** (The `PyAudio` task). Refactor the `AudioRecorder` to process audio in small chunks and implement a basic VAD (Voice Activity Detection) to detect the end of speech.
