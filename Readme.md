# voiceChain: A Local-First, Streaming, and Interruptible Voice Agent Framework

## Introduction

voiceChain is a high-performance, modular Python library for building real-time, voice-based conversational AI agents. Its core mission is to provide a complete, **local-first** framework that runs entirely offline on consumer hardware, with a special focus on leveraging the full power of Apple Silicon via the MLX and `llama.cpp` Metal backends. The novel architecture is designed from the ground up for low-latency streaming and full user interruptibility, creating a truly responsive conversational experience.

## Core Features

*   **End-to-End Local Pipeline:** All components—including Speech-to-Text (Whisper), Large Language Model (Llama/Qwen), and Text-to-Speech (Kokoro)—run 100% offline, ensuring privacy and independence from cloud services.
*   **High-Performance on Apple Silicon:** Leverages Apple's Metal framework via MLX and `llama-cpp-python` for massive GPU acceleration on all AI model inferences.
*   **Low-Latency Streaming:** A fully asynchronous design using `asyncio` allows for the parallel processing of STT, LLM inference, and TTS synthesis. This minimizes "dead air" and begins audio playback as soon as the first sentence is generated.
*   **Hands-Free Operation:** A sophisticated two-stage Voice Activity Detection (VAD) system (WebRTCVAD + Silero VAD) provides robust, continuous listening for a "wake-word free" experience.
*   **Full Interruptibility (Barge-In):** Users can interrupt the agent at any point during its response. The pipeline gracefully cancels the in-flight generation and playback, immediately processing the user's new command.
*   **Software-Based Echo Cancellation:** A pragmatic, text-based echo detection algorithm prevents the agent from accidentally transcribing and responding to its own speech, a common issue in full-duplex systems.

## Architecture Overview

voiceChain is built on a clean, decoupled architecture that separates concerns for maintainability and scalability.

*   **ServiceManager:** Manages all background hardware interactions, I/O, and threading. This includes the microphone input thread, the VAD processor, the persistent audio output stream, and dedicated thread pools for STT and TTS tasks.
*   **PipelineRunner:** Orchestrates a single, complete conversational "turn" from user audio to agent audio response. It manages the flow of data through the STT, LLM, and TTS models.
*   **ConversationManager:** The main state machine of the application. It listens for user speech from the `ServiceManager` and decides when to initiate a new turn with the `PipelineRunner`, when to handle a barge-in, and when to return to an idle listening state.
*   **Composition Root:** The `examples/run_agent.py` script acts as the composition root, where all components are instantiated with their dependencies and the application is started.

## Getting Started

### Prerequisites

*   Python 3.10+
*   **ffmpeg:** Required by `mlx-whisper` for audio processing.
    ```bash
    # On macOS with Homebrew
    brew install ffmpeg
    ```
*   **PortAudio:** Required by `PyAudio` and `sounddevice` for microphone and speaker access.
    ```bash
    # On macOS with Homebrew
    brew install portaudio
    ```

### Installation

1.  Clone the repository and navigate to the root directory.
2.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the library in "editable" mode, which also installs all dependencies from `pyproject.toml`:
    ```bash
    pip install -e .
    ```

### Model Setup

The agent requires several pre-trained models to function. You must download them and place them in a `models/` directory at the root of the project.

```
project_root/
├── models/
│   ├── whisper-large-v3-turbo/      # MLX Whisper model
│   ├── Qwen3-4B-Instruct-2507-Q4_K_M.gguf  # GGUF format LLM
│   └── Kokoro/                    # MLX Kokoro TTS model
└── ...
```

### Running the Agent

Once the environment is set up and the models are in place, you can run the agent with a single command from the project root:

```bash
python examples/run_agent.py
```

## Project Structure

The library uses a modern `src` layout to cleanly separate library code from tests and examples.

```
src/
└── voiceChain/
    ├── __init__.py
    ├── audio/          # Hardware I/O: Player, Recorder, VAD
    │   ├── player.py
    │   ├── recorder.py
    │   └── vad.py
    ├── core/           # Core AI Engines: LLM, STT, TTS
    │   ├── llm.py
    │   ├── stt.py
    │   └── tts.py
    ├── pipeline/       # Orchestration and State Management
    │   ├── manager.py
    │   ├── runner.py
    │   └── services.py
    └── utils/          # Shared utilities (Logging, State Enums)
        ├── logging.py
        └── state.py
```
```

