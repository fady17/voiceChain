import asyncio
import numpy as np
import sounddevice as sd
from loguru import logger

class AudioPlayer:
    """
    Manages audio playback in a non-blocking way using an asyncio queue and a
    persistent sounddevice output stream.
    """
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self._playback_queue = asyncio.Queue()
        self._playback_task = None
        self._stream = None
        logger.info("AudioPlayer initialized for persistent stream playback.")

    def start(self):
        """
        Starts the audio output stream and the async playback loop.
        The stream is "primed" with silence to reduce latency on the first playback.
        """
        try:
            self._stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32')
            self._stream.start()
            # Prime the stream buffer to reduce latency on first play
            prime_buffer = np.zeros(1024, dtype=np.float32)
            self._stream.write(prime_buffer)
            logger.info("AudioPlayer stream primed with silence.")
            self._playback_task = asyncio.create_task(self._playback_loop())
            logger.success("AudioPlayer stream started and playback loop is running.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to open audio stream. Error: {e}")

    async def stop(self):
        """Stops the playback loop and closes the audio stream."""
        if self._playback_task:
            self._playback_task.cancel()
        if self._stream:
            self._stream.stop()
            self._stream.close()
        logger.success("AudioPlayer stream stopped and closed.")

    async def add_to_queue(self, audio_data: np.ndarray):
        """Adds a chunk of audio data to the playback queue."""
        if audio_data is not None and audio_data.size > 0:
            await self._playback_queue.put(audio_data)

    async def interrupt(self):
        """
        Immediately stops playback and clears the queue. Essential for barge-in.
        """
        logger.warning("AudioPlayer received interrupt signal!")
        # Clear the queue
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("Playback queue cleared.")
        # Stop the physical stream immediately
        sd.stop()
        logger.info("Audio stream stopped immediately.")

    async def _playback_loop(self):
        """The core async task that pulls audio from the queue and writes it to the stream."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                audio_data = await self._playback_queue.get()
                if audio_data.ndim > 1:
                    audio_data = audio_data.ravel() # Ensure 1D array
                
                # ASYNC INTEGRITY: The blocking stream.write call is offloaded to an
                # executor, keeping the main event loop responsive.
                if self._stream:
                    await loop.run_in_executor(None, self._stream.write, audio_data)
                logger.success("Player loop finished writing audio chunk.")
        except asyncio.QueueEmpty:
            pass # This is normal if queue is empty
        except asyncio.CancelledError:
            logger.info("Playback loop cancelled.")
        except Exception as e:
            logger.error(f"Error in playback loop: {e}", exc_info=True)