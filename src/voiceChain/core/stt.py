import time
import numpy as np
import mlx_whisper
from loguru import logger

class Transcriber:
    """
    A synchronous worker for transcribing audio using MLX Whisper.
    Designed to be run in a thread pool executor to avoid blocking the main async loop.
    Model is pre-loaded at initialization.
    """
    def __init__(self, model_path: str = "models/whisper-large-v3-turbo"):
        """
        Initializes the Transcriber and pre-loads the whisper model.
        This can take some time and memory.
        """
        self.model_path = model_path
        logger.info("Transcriber worker initialized. Model will be loaded on first use by mlx_whisper.")
        # Note: mlx_whisper implicitly handles model loading on the first call,
        # which is functionally equivalent to pre-loading in this single-threaded worker context.

    def transcribe_audio_sync(self, audio_data: np.ndarray) -> str | None:
        """
        Performs synchronous audio transcription.

        Args:
            audio_data: A NumPy array containing the audio waveform.

        Returns:
            The transcribed text as a string, or None if transcription fails.
        """
        logger.info("Starting synchronous transcription...")
        start_time = time.time()
        try:
            # PERFORMANCE: This call leverages MLX for GPU acceleration on Apple Silicon.
            result = mlx_whisper.transcribe(audio=audio_data, path_or_hf_repo=self.model_path, language="en")
            transcribed_text = result["text"].strip() if result else None
        except Exception as e:
            logger.error(f"Transcription failed in worker thread: {e}", exc_info=True)
            return None
        elapsed_time = time.time() - start_time
        logger.success(f"Transcription complete in {elapsed_time:.2f}s. Text: '{transcribed_text}'")
        return transcribed_text