import threading
import queue
import pyaudio
from loguru import logger

def audio_input_thread_worker(audio_queue: queue.Queue, is_running_event: threading.Event, sample_rate=16000, frame_duration_ms=30):
    """
    A dedicated thread worker that captures audio from the microphone using PyAudio
    and puts raw frames into a thread-safe queue.
    """
    logger.info("Audio input worker thread started.")
    p = pyaudio.PyAudio()
    chunk_size = int(sample_rate * frame_duration_ms / 1000)
    stream = None
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)
        logger.success("PyAudio stream opened. Now reading frames.")
        while is_running_event.is_set():
            frame = stream.read(chunk_size, exception_on_overflow=False)
            try:
                audio_queue.put(frame, block=False)
            except queue.Full:
                logger.warning("Audio input queue is full, dropping a frame.")
    except Exception as e:
        logger.error(f"Error in audio input thread: {e}", exc_info=True)
    finally:
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()
        logger.info("PyAudio stream resources released.")