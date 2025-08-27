---
**Entry: 2024-05-23**
**Topic:** Selection and Validation of the Initial TTS Engine
**Status:** Decision Made

**1. Context:**
To complete Phase 0, we needed to select and validate a Text-to-Speech (TTS) engine that could run locally on the M4 Mac via the MLX framework. The primary requirements were performance (low RTF for streaming potential) and quality.

**2. Exploration & Alternatives:**
*   **Alternative 1: XTTS-v2.** A high-quality model, but initial investigation suggested a more complex API and potentially higher computational cost.
*   **Alternative 2: Kokoro-82M.** A smaller, more modern model specifically available in the `mlx-community`. The documentation trail was complex, leading us from manual implementation attempts to the discovery of the `mlx-audio` library.
*   **Experiment:** We ran the `test_tts.py` script against the local Kokoro model.
*   **Key Result:** The engine's verbose output reported a **Real-time Factor (RTF) of 0.12x**, meaning it can synthesize audio ~8.3 times faster than real-time. Peak memory was a reasonable 2.23GB.

**3. Decision & Rationale:**
We have selected **Kokoro-82M**, executed via the `mlx-audio` library, as our baseline TTS engine.
*   **Rationale:** The exceptionally low RTF directly serves our Guiding Principle of "Minimize Perceived Latency." Its availability as a well-supported MLX community model ensures maintainability. The quality is sufficient for our initial pipeline, and its low resource footprint leaves performance budget for other components.

**4. Next Steps & Implications:**
*   We will integrate the `mlx_audio.tts.generate` function into our `pipeline_v1.py` script.
*   The function's side-effect nature (saving a file instead of returning a waveform) will need to be managed in the pipeline's control flow.

---