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



*   **Phase 3A: The Software-First State Machine.** We perfect all turn-taking logic using explicit, software-based triggers.
*   **Phase 3B: Real-Time Audio & VAD Integration.** We swap our software triggers for a real-time audio stream and a VAD, now with a robust state machine ready to handle its output.


#### **Phase 3A: Building the Core Conversational State Machine**

**Objective:** To create a  state-driven agent that understands concepts like `LISTENING`, `THINKING`, `SPEAKING`, and can handle interruptions, using simple keyboard inputs as triggers instead of a live microphone.

*   **Task 3A.1: Architect the State Machine.**
    *   **Action:** Formally define the agent's states (`IDLE`, `LISTENING`, `PROCESSING`, `SPEAKING`, `WAITING_FOR_USER`). We will draw this out and define the exact conditions that trigger transitions between states (e.g., transition from `SPEAKING` to `LISTENING` if a "barge-in" event occurs).
    *   **Tool:** We can use a simple Enum for the states. This is a pure software design task.

*   **Task 3A.2: Implement the "Push-to-Talk" Trigger.**
    *   **Action:** Replace `input("Press Enter to start recording...")` with a more interactive loop. We'll use a library like `pynput` or `keyboard` to detect a key press and hold.
    *   **User Experience:** "Hold the spacebar to talk."
    *   **State Transition:** `IDLE` -> `key_press` -> `LISTENING`. `LISTENING` -> `key_release` -> `PROCESSING`.
    *   **Benefit:** This gives us fine-grained control over the start and end of an utterance, perfectly simulating what a VAD would do, but without any of the acoustic complexity.

*   **Task 3A.3: Implement Software-Based "Barge-In".**
    *   **Action:** While the agent is in the `SPEAKING` state (i.e., our `AudioPlayer` is active), we will listen for another key press (e.g., the spacebar again).
    *   **State Transition:** `SPEAKING` -> `key_press` -> `LISTENING`.
    *   **Core Logic:** When this transition occurs, the orchestrator must immediately:
        1.  Send a signal to the `AudioPlayer` to stop the current playback.
        2.  Cancel any pending TTS tasks.
        3.  Clear any buffered text from the LLM stream.
        4.  Begin recording the new user utterance.
    *   **Benefit:** We will build and perfect the entire complex interruption logic in a 100% reproducible software environment.

*   **Task 3A.4: Refactor the `AudioPlayer` for Interruptibility.**
    *   **Action:** Our current `AudioPlayer` is not designed to be stopped mid-playback. We will modify it to support an `stop_current_playback()` method. This will likely involve using an `asyncio.Event` that the playback loop can check.
    *   **Benefit:** This creates a crucial, reusable software primitive for controlling the agent's voice.

---

#### **Phase 3B: Integrating Real-World Audio**

**Objective:** To replace our software-based triggers (`Push-to-Talk`, `Barge-In Key Press`) with a live, continuous audio stream and a real VAD.

*   **Task 3B.1: Implement Continuous Audio Streaming.**
    *   **Action:** Refactor `AudioRecorder` to use a non-blocking stream (now is the time for `PyAudio` or `sounddevice`'s stream API). It will continuously read small chunks of audio from the microphone and place them into an `asyncio.Queue`.

*   **Task 3B.2: Integrate a VAD Model.**
    *   **Action:** Create a new `VADProcessor` task. It will consume audio chunks from the queue created in 3B.1. We'll use a lightweight, high-performance VAD like `silero-vad`.
    *   **Output:** The VAD task will not output audio; it will output *events*: `SPEECH_STARTED`, `SPEECH_ENDED`.

*   **Task 3B.3: Connect VAD Events to the State Machine.**
    *   **Action:** This is the final, elegant step. We replace our keyboard listeners from Phase 3A.
    *   **State Transition:**
        *   `IDLE` -> `SPEECH_STARTED` event -> `LISTENING`.
        *   `LISTENING` -> `SPEECH_ENDED` event -> `PROCESSING`.
        *   `SPEAKING` -> `SPEECH_STARTED` event -> `BARGE_IN` -> `LISTENING`.


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


*   **Task 3B.4: The Acoustic Echo Problem (Now a Manageable Task).**
    *   **Action:** At this point, the system will work perfectly *with a headset*. Without one, it will hear itself and barge-in constantly. NOW we can tackle this.
    *   **Solutions (to be explored):**
        1.  **Software AEC:** Integrate a software-based echo cancellation library (e.g., `webrtc-audio-processing`, `speexdsp`).
        2.  **"Duck" and Mute:** A simpler approach. When the agent is `SPEAKING`, we can programmatically mute the microphone input or instruct the VAD to ignore any detected speech. This is less elegant but highly effective.


The Path Forward: Phase 4 - The Production-Grade Refactor
This architectural review gives us a crystal-clear roadmap for our next phase. We will stop adding new features and focus on hardening and optimizing the incredible system we've already built.
Phase 4 Checklist:

Task 4.1: In-Memory Audio Streaming.
Action: Refactor the tts_consumer and AudioPlayer to pass audio data as in-memory NumPy arrays, completely eliminating the TTS .wav file I/O. (This should be our very next task).

Task 4.2: Atomic State Transitions.
Action: Introduce an asyncio.Lock to the ConversationManager and protect all self.state modifications.

Task 4.3: Robust Error Handling.
Action: Implement more granular error handling within the run_pipeline method with specific verbal error messages.

Task 4.4: Resource Cleanup.
Action: If we stick with any file I/O, ensure temporary files are deleted. Implement robust signal handling for graceful shutdown.

Task 4.5 (Research Spike): Single VAD System.
Action: Create a branch to experiment with removing WebRTCVAD and using only Silero. Measure CPU impact and determine if the simplification is worth it.

Task 4.6 (Research Spike): Advanced Barge-In.
Action: Brainstorm and prototype a more advanced is_echo function, potentially using acoustic features.

### TODO:

## Speech-to-Text (STT) Engine
Streaming Transcription : For real-time feedback, a streaming model that provides partial transcripts as the user speaks is the gold standard. This dramatically improves the perception of speed.
Metadata Generation: This is where we can innovate. The STT shouldn't just output text. It could also output:
Word-level timestamps: Crucial for understanding timing and for enabling features like real-time visual feedback.
Acoustic Embeddings/Prosody Features: Capturing the way something was said (tone, pitch, energy). This data is invaluable for the LLM and TTS to generate a more contextually appropriate response.
* real-Time Transcription: 
    * Generating the response token by token, rather than all at once. This allows the TTS to start speaking before the LLM has finished its entire thought, drastically reducing perceived latency.


* Noise Reduction & Normalization: Cleaning the audio signal before it hits the STT engine to improve accuracy.
Speaker Output Management: Handling the playback of the synthesized TTS audio, including managing potential overlaps or interruptions.


# the Core Orchestrator & State Manager
State Machine Management: Tracking the system's state (e.g., LISTENING, THINKING, SPEAKING, IDLE).
Turn-taking and Interruption Handling: Deciding when the LLM should process input and, critically, allowing the user to interrupt the TTS playback (barge-in). This is a hallmark of a natural-feeling system.
Context Aggregation: It gathers information from all components—the text from STT, the emotional cue from its acoustic metadata, the conversation history—and formats it into a coherent prompt for the LLM.
Dispatching Commands: Directing the LLM output to the TTS engine or potentially other system functions (e.g., running a script, calling an API).

# Large Language Model (LLM) Engine
Generating the response token by token, rather than all at once. This allows the TTS to start speaking before the LLM has finished its entire thought, drastically reducing perceived latency.

# Text-to-Speech (TTS) Engine
* The TTS engine should be able to take the original prosody metadata from the user's speech (via the STT and Orchestrator) and mirror it in its own output. If the user sounds inquisitive, the response should sound inquisitive. This creates an empathetic feedback loop. It could also take explicit prosody instructions from the LLM (e.g., [start_excited] That's a great idea! [end_excited]).

