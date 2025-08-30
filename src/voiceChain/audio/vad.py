import asyncio
import sys
import numpy as np
import torch
import webrtcvad
import collections
from loguru import logger

def load_silero_vad(force_reload: bool = False):
    """
    Loads the Silero VAD model from torch.hub.
    This is a one-time setup operation.
    """
    logger.info("Loading Silero VAD model from torch.hub...")
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=force_reload,
            trust_repo=True
        )
        logger.success("Silero VAD model and utilities loaded successfully.")
        return model, utils
    except Exception as e:
        logger.critical(f"Failed to load Silero VAD model: {e}")
        sys.exit(1)

class VADProcessor:
    """
    Processes a raw audio stream using a two-stage VAD (WebRTCVAD -> Silero VAD)
    to detect and emit complete user utterances.
    """
    def __init__(self, raw_audio_queue: asyncio.Queue, utterance_queue: asyncio.Queue, silero_model, silero_utils, sample_rate=16000, frame_duration_ms=30, padding_ms=300):
        self.raw_audio_queue = raw_audio_queue
        self.utterance_queue = utterance_queue
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        num_padding_frames = padding_ms // frame_duration_ms
        self.ring_buffer = collections.deque(maxlen=num_padding_frames)
        self.ratio = 0.75
        self.silero_model = silero_model
        self.get_speech_timestamps = silero_utils[0]
        self.vad_task = None

    def is_speech(self, frame: bytes) -> bool:
        return self.vad.is_speech(frame, self.sample_rate)

    async def start(self):
        logger.info("Starting Two-Stage VAD processor...")
        self.vad_task = asyncio.create_task(self._vad_collector())

    async def stop(self):
        if self.vad_task:
            self.vad_task.cancel()
            await asyncio.sleep(0.1) # Allow cancellation to propagate
        logger.success("VAD processor stopped.")

    async def _vad_collector(self):
        triggered = False
        voiced_frames = []
        try:
            while True:
                frame = await self.raw_audio_queue.get()
                is_speech_frame = self.is_speech(frame)

                if not triggered:
                    self.ring_buffer.append((frame, is_speech_frame))
                    num_voiced = len([f for f, s in self.ring_buffer if s])
                    if num_voiced > self.ratio * self.ring_buffer.maxlen:
                        logger.debug("WebRTCVAD triggered.")
                        triggered = True
                        voiced_frames.extend(f for f, s in self.ring_buffer)
                        self.ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    self.ring_buffer.append((frame, is_speech_frame))
                    num_unvoiced = len([f for f, s in self.ring_buffer if not s])
                    if num_unvoiced > self.ratio * self.ring_buffer.maxlen:
                        logger.debug("WebRTCVAD un-triggered.")
                        triggered = False
                        await self.confirm_with_silero(b''.join(voiced_frames))
                        self.ring_buffer.clear()
                        voiced_frames = []
        except asyncio.CancelledError:
            logger.info("VAD collector task cancelled.")

    async def confirm_with_silero(self, audio_bytes: bytes):
        logger.info("WebRTCVAD detected utterance, confirming with Silero VAD...")
        audio_float32 = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float32)
        loop = asyncio.get_running_loop()
        
        # ASYNC INTEGRITY: The synchronous Silero model call is run in an executor
        # to prevent blocking the event loop, preserving the original async pattern.
        speech_timestamps = await loop.run_in_executor(
            None, 
            lambda: self.get_speech_timestamps(
                audio_tensor, self.silero_model, sampling_rate=self.sample_rate
            )
        )
        
        if len(speech_timestamps) > 0:
            logger.success("Silero VAD confirmed speech. Emitting utterance.")
            await self.utterance_queue.put(audio_float32)
        else:
            logger.warning("Silero VAD detected NOISE. Discarding utterance.")