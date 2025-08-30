import asyncio
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from ..audio.recorder import audio_input_thread_worker
from ..audio.vad import VADProcessor, load_silero_vad
from ..audio.player import AudioPlayer

class ServiceManager:
    """

    Manages all background services required for the conversation pipeline,
    including audio I/O, VAD, and dedicated thread pools for ML tasks.
    """
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.audio_input_queue = queue.Queue(maxsize=100)
        self.raw_audio_queue = asyncio.Queue(maxsize=100)
        self.user_utterance_queue = asyncio.Queue()
        self.is_audio_running = threading.Event()
        self.audio_input_thread = None
        self.audio_poller_task = None
        
        # Initialize dependencies
        silero_model, silero_utils = load_silero_vad()
        self.vad_processor = VADProcessor(self.raw_audio_queue, self.user_utterance_queue, silero_model, silero_utils)
        self.player = AudioPlayer()
        
        self._create_executors()

    def _create_executors(self):
        """
        Creates dedicated single-thread executors for STT and TTS to ensure
        sequential processing and preserve MLX Metal performance.
        """
        logger.info("Creating dedicated thread pool executors for STT and TTS.")
        self.stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stt_worker')
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tts_worker')

    async def _audio_poller(self):
        """
        An async task that polls the thread-safe audio input queue and puts
        frames into the async-safe queue for the VAD processor.
        This bridges the synchronous threading world with the asyncio world.
        """
        logger.info("Audio poller task started.")
        loop = asyncio.get_running_loop()
        try:
            while self.is_audio_running.is_set():
                frame = await loop.run_in_executor(None, self.audio_input_queue.get)
                await self.raw_audio_queue.put(frame)
        except asyncio.CancelledError:
            logger.info("Audio poller task cancelled.")
        except Exception as e:
            logger.error(f"Error in audio poller: {e}")

    async def start(self):
        """Starts all managed services."""
        self.player.start()
        self.is_audio_running.set()
        self.audio_input_thread = threading.Thread(
            target=audio_input_thread_worker,
            args=(self.audio_input_queue, self.is_audio_running),
            daemon=True
        )
        self.audio_input_thread.start()
        self.audio_poller_task = asyncio.create_task(self._audio_poller())
        await self.vad_processor.start()

    async def stop(self):
        """Stops all managed services gracefully."""
        logger.info("Stopping agent services...")
        self.is_audio_running.clear()
        if self.audio_poller_task:
            self.audio_poller_task.cancel()
        if self.audio_input_thread and self.audio_input_thread.is_alive():
            self.audio_input_thread.join(timeout=2)
        await self.vad_processor.stop()
        await self.player.stop()
        self._shutdown_executors()
        
    def _shutdown_executors(self):
        logger.info("Shutting down thread pool executors...")
        self.stt_executor.shutdown(wait=True)
        self.tts_executor.shutdown(wait=True)

    def reset_executors(self):
        """
        Forcefully shuts down and recreates executors. Necessary for handling
        barge-in interrupts to cancel in-flight STT/TTS tasks.
        """
        logger.warning("Resetting STT and TTS executors...")
        # Use cancel_futures=True for immediate termination of tasks
        self.stt_executor.shutdown(wait=False, cancel_futures=True)
        self.tts_executor.shutdown(wait=False, cancel_futures=True)
        self._create_executors()