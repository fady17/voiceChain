import time
import numpy as np
from pathlib import Path
from loguru import logger
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.models.kokoro import KokoroPipeline

class TextToSpeechEngine:
    """
    A stateful, synchronous worker that pre-loads the TTS pipeline
    at initialization. Designed to be run in a thread pool executor.
    """
    MODEL_REPO_ID = "mlx-community/Kokoro-82M-bf16"

    def __init__(self, model_path: str = "models/Kokoro", speaker: str = "af_heart"):
        """
        Initializes the TTS engine and pre-loads the Kokoro model and pipeline.
        This is a critical step for performance, ensuring the model is ready in memory.
        """
        self.model_path = model_path
        self.speaker = speaker
        self.pipeline = None
        
        logger.info(f"Initializing and pre-loading TTS Pipeline from local path: '{self.model_path}'...")
        try:
            # CRITICAL: Model preloading sequence. This uses the library's high-level
            # utility to correctly instantiate the model from config files, then
            # initializes the pipeline. This preserves the original script's behavior.
            model = load_model(self.model_path)
            logger.success("Core Kokoro MLX model loaded successfully via utility.")
            
            self.pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=self.MODEL_REPO_ID)
            logger.success("Kokoro TTS Pipeline pre-loaded and ready.")

        except Exception as e:
            logger.critical(f"CRITICAL: Failed to pre-load TTS Pipeline. Error: {e}", exc_info=True)
            self.pipeline = None

    def synthesize_speech_sync(self, text_chunk: str) -> np.ndarray | None:
        """
        Synthesizes text using the pre-loaded pipeline and returns the audio
        waveform as a NumPy array. This is a synchronous, blocking call.
        """
        if not self.pipeline:
            logger.error("TTS Pipeline not available.") 
            return None
        
        logger.info(f"Starting in-memory synthesis for chunk: '{text_chunk}'")
        start_time = time.time()
        
        try:
            local_voice_path = str(Path(self.model_path) / f"{self.speaker}.pt")
            
            # PERFORMANCE: This generator-based synthesis leverages MLX for GPU acceleration.
            results_generator = self.pipeline(text=text_chunk, voice=local_voice_path)
            
            audio_chunks = [np.array(audio, copy=False) for _, _, audio in results_generator if audio is not None]

            if not audio_chunks:
                logger.warning("TTS synthesis produced no audio data.") 
                return None
            
            audio_data = np.concatenate(audio_chunks)
            elapsed_time = time.time() - start_time
            logger.success(f"In-memory synthesis complete in {elapsed_time:.2f}s.")
            return audio_data
            
        except Exception as e:
            logger.error(f"Exception during in-memory synthesis: {e}", exc_info=True)
            return None