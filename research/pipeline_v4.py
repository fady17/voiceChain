import asyncio
import threading
import sys
import time
import numpy as np
from scipy.io.wavfile import read
from llama_cpp import Llama
import mlx_whisper
from enum import Enum, auto
from loguru import logger
import pyaudio
import webrtcvad
import collections
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import soundfile as sf
import queue
import mlx.core as mx
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.models.kokoro import Model as KokoroModel
import sounddevice as sd

# --- OBSERVABILITY SETUP ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/pipeline_v4_{time}.json", level="DEBUG", serialize=True, rotation="10 MB", catch=True)
logger.info("Logger configured. Starting Stateful Voice Agent.")

# --- VAD MODEL AND UTILITIES (LOADED ONCE AT STARTUP) ---
logger.info("Loading Silero VAD model from torch.hub...")
try:
    silero_model, utils_tuple = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = utils_tuple
    logger.success("Silero VAD model and utilities loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load Silero VAD model: {e}"); sys.exit(1)

# ==================================================================================
# SECTION 1: SYNCHRONOUS WORKER CLASSES
# ==================================================================================

class Transcriber:
    def __init__(self, model_path="models/whisper-large-v3-turbo"):
        self.model_path = model_path
        logger.info(f"Transcriber worker initialized with model: '{self.model_path}'.")

    def transcribe_audio_sync(self, audio_data: np.ndarray) -> str | None:
        logger.info("Starting synchronous transcription...")
        start_time = time.time()
        try:
            result = mlx_whisper.transcribe(audio=audio_data, path_or_hf_repo=self.model_path, language="en")
            transcribed_text = result["text"].strip() if result else None
        except Exception as e:
            logger.error(f"Transcription failed in worker thread: {e}", exc_info=True); return None
        elapsed_time = time.time() - start_time
        logger.success(f"Transcription complete in {elapsed_time:.2f}s. Text: '{transcribed_text}'")
        return transcribed_text

class TextToSpeechEngine:
    """
    A stateful, synchronous worker that pre-loads the TTS pipeline
    at initialization by using the library's robust `load_model` utility.
    """
    LOCAL_MODEL_PATH = "models/Kokoro"
    MODEL_REPO_ID = "mlx-community/Kokoro-82M-bf16"
    
    def __init__(self, speaker="af_heart"):
        self.SPEAKER = speaker
        logger.info(f"Initializing and pre-loading TTS Pipeline from local path: '{self.LOCAL_MODEL_PATH}'...")
        
        try:
            # --- THE DEFINITIVE FIX ---
            # Use the official, high-level utility to load the model.
            # This function correctly handles the config.json and model instantiation.
            model = load_model(self.LOCAL_MODEL_PATH)
            logger.success("Core Kokoro MLX model loaded successfully via utility.")
            
            # Now, initialize the pipeline with the successfully loaded model.
            self.pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=self.MODEL_REPO_ID)
            logger.success("Kokoro TTS Pipeline pre-loaded and ready.")

        except Exception as e:
            logger.critical(f"CRITICAL: Failed to pre-load TTS Pipeline. Error: {e}", exc_info=True)
            self.pipeline = None
            
    def synthesize_speech_sync(self, text_chunk: str) -> np.ndarray | None:
        """
        Synthesizes text using the pre-loaded pipeline and returns the audio
        waveform as a NumPy array.
        """
        if not self.pipeline:
            logger.error("TTS Pipeline not available."); return None
        
        logger.info(f"Starting in-memory synthesis for chunk: '{text_chunk}'")
        start_time = time.time()
        
        try:
            local_voice_path = str(Path(self.LOCAL_MODEL_PATH) / f"{self.SPEAKER}.pt")
            results_generator = self.pipeline(text=text_chunk, voice=local_voice_path)
            
            audio_chunks = [np.array(audio, copy=False) for _, _, audio in results_generator if audio is not None]

            if not audio_chunks:
                logger.warning("TTS synthesis produced no audio data."); return None
            
            audio_data = np.concatenate(audio_chunks)
            elapsed_time = time.time() - start_time
            logger.success(f"In-memory synthesis complete in {elapsed_time:.2f}s.")
            return audio_data
            
        except Exception as e:
            logger.error(f"Exception during in-memory synthesis: {e}", exc_info=True)
            return None
# ==================================================================================
# SECTION 2: BACKGROUND HARDWARE/SERVICE CLASSES
# ==================================================================================

def audio_input_thread_worker(audio_queue: queue.Queue, is_running_event: threading.Event, sample_rate=16000, frame_duration_ms=30):
    logger.info("Audio input worker thread started.")
    p = pyaudio.PyAudio()
    chunk_size = int(sample_rate * frame_duration_ms / 1000)
    stream = None
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)
        logger.success("PyAudio stream opened. Now reading frames.")
        while is_running_event.is_set():
            frame = stream.read(chunk_size, exception_on_overflow=False)
            try: audio_queue.put(frame, block=False)
            except queue.Full: logger.warning("Audio input queue is full, dropping a frame.")
    except Exception as e:
        logger.error(f"Error in audio input thread: {e}", exc_info=True)
    finally:
        if stream and stream.is_active(): stream.stop_stream(); stream.close()
        p.terminate()
        logger.info("PyAudio stream resources released.")

class VADProcessor:
    def __init__(self, raw_audio_queue: asyncio.Queue, utterance_queue: asyncio.Queue, sample_rate=16000, frame_duration_ms=30, padding_ms=300):
        self.raw_audio_queue = raw_audio_queue; self.utterance_queue = utterance_queue
        self.sample_rate = sample_rate; self.frame_duration_ms = frame_duration_ms
        self.vad = webrtcvad.Vad(3); num_padding_frames = padding_ms // frame_duration_ms
        self.ring_buffer = collections.deque(maxlen=num_padding_frames)
        self.ratio = 0.75; self.silero_model = silero_model; self.vad_task = None

    def is_speech(self, frame: bytes) -> bool: return self.vad.is_speech(frame, self.sample_rate)

    async def start(self):
        logger.info("Starting Two-Stage VAD processor..."); self.vad_task = asyncio.create_task(self._vad_collector())

    async def stop(self):
        if self.vad_task: self.vad_task.cancel(); await asyncio.sleep(0.1)
        logger.success("VAD processor stopped.")

    async def _vad_collector(self):
        triggered = False; voiced_frames = []
        while True:
            try:
                frame = await self.raw_audio_queue.get()
                is_speech = self.is_speech(frame)
                if not triggered:
                    self.ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, s in self.ring_buffer if s])
                    if num_voiced > self.ratio * self.ring_buffer.maxlen:
                        logger.debug("WebRTCVAD triggered."); triggered = True
                        for f, s in self.ring_buffer: voiced_frames.append(f)
                        self.ring_buffer.clear()
                else:
                    voiced_frames.append(frame); self.ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, s in self.ring_buffer if not s])
                    if num_unvoiced > self.ratio * self.ring_buffer.maxlen:
                        logger.debug("WebRTCVAD un-triggered."); triggered = False
                        await self.confirm_with_silero(b''.join(voiced_frames))
                        self.ring_buffer.clear(); voiced_frames = []
            except asyncio.CancelledError: break

    async def confirm_with_silero(self, audio_bytes: bytes):
        logger.info("WebRTCVAD detected utterance, confirming with Silero VAD...")
        audio_float32 = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float32)
        loop = asyncio.get_running_loop()
        speech_timestamps = await loop.run_in_executor(
            None, lambda: get_speech_timestamps(audio_tensor, self.silero_model, sampling_rate=self.sample_rate)
        )
        if len(speech_timestamps) > 0:
            logger.success("Silero VAD confirmed speech. Emitting utterance."); await self.utterance_queue.put(audio_float32)
        else:
            logger.warning("Silero VAD detected NOISE. Discarding utterance.")

class AudioPlayer:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate; self._playback_queue = asyncio.Queue(); self._playback_task = None; self._stream = None
        logger.info("AudioPlayer initialized for persistent stream playback.")

    def start(self):
        try:
            self._stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32')
            self._stream.start()
            prime_buffer = np.zeros(1024, dtype=np.float32); self._stream.write(prime_buffer)
            logger.info("AudioPlayer stream primed with silence.")
            self._playback_task = asyncio.create_task(self._playback_loop())
            logger.success("AudioPlayer stream started and playback loop is running.")
        except Exception as e: logger.critical(f"CRITICAL: Failed to open audio stream. Error: {e}")

    async def stop(self):
        if self._playback_task: self._playback_task.cancel()
        if self._stream: self._stream.stop(); self._stream.close()
        logger.success("AudioPlayer stream stopped and closed.")

    async def add_to_queue(self, audio_data: np.ndarray):
        if audio_data is not None and audio_data.size > 0: await self._playback_queue.put(audio_data)

    async def interrupt(self):
        logger.warning("AudioPlayer received interrupt signal!")
        while not self._playback_queue.empty():
            try: self._playback_queue.get_nowait()
            except asyncio.QueueEmpty: break
        logger.info("Playback queue cleared."); sd.stop(); logger.info("Audio stream stopped immediately.")

    async def _playback_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                audio_data = await self._playback_queue.get()
                if audio_data.ndim > 1: audio_data = audio_data.ravel()
                if self._stream: await loop.run_in_executor(None, self._stream.write, audio_data)
                logger.success(f"Player loop finished writing audio chunk.")
            except asyncio.QueueEmpty: continue
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"Error in playback loop: {e}", exc_info=True)

# ==================================================================================
# SECTION 3: ASYNC ORCHESTRATION CLASSES
# ==================================================================================

class AgentState(Enum):
    IDLE = auto(); PROCESSING = auto(); RESPONDING = auto()

class LLMEngine:
    def __init__(self, llm_instance):
        self.llm = llm_instance; logger.success("LLMEngine initialized with pre-loaded model.")

    async def generate_response(self, user_prompt: str, text_queue: asyncio.Queue):
        prompt_messages = [
            {"role": "system", "content": "You are a helpful, brief, and conversational AI assistant. Your responses must be broken into complete sentences, ending with proper punctuation. Do not use emojis."},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(f"Starting LLM stream for prompt: '{user_prompt}'")
        start_time = time.time(); loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(None, lambda: self.llm.create_chat_completion(messages=prompt_messages, max_tokens=150, stream=True))
        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta: await text_queue.put(delta['content'])
        await text_queue.put(None)
        logger.success(f"LLM stream finished in {time.time() - start_time:.2f}s.")

class PipelineRunner:
    def __init__(self, transcriber: Transcriber, llm_engine: LLMEngine, tts_engine: TextToSpeechEngine, services: 'ServiceManager'):
        self.transcriber = transcriber; self.llm_engine = llm_engine
        self.tts_engine = tts_engine; self.services = services

    async def run(self, audio_data: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        transcribed_text = await loop.run_in_executor(
            self.services.stt_executor, self.transcriber.transcribe_audio_sync, audio_data
        )
        if not transcribed_text: raise ValueError("Transcription failed.")
        
        text_token_queue = asyncio.Queue(); tts_sentence_queue = asyncio.Queue(); full_text_future = asyncio.Future()
        
        llm_task = asyncio.create_task(self.llm_engine.generate_response(transcribed_text, text_token_queue))
        chunker_task = asyncio.create_task(self.text_chunker(text_token_queue, tts_sentence_queue, full_text_future))
        tts_task = asyncio.create_task(self.tts_consumer(tts_sentence_queue))
        
        await asyncio.gather(llm_task, chunker_task, tts_task)
        return await full_text_future

    async def text_chunker(self, text_queue: asyncio.Queue, tts_queue: asyncio.Queue, full_text_future: asyncio.Future):
        sentence_buffer = ""; full_response_text = ""
        sentence_terminators = {".", "?", "!"}
        while True:
            token = await text_queue.get()
            if token is None:
                if sentence_buffer.strip(): await tts_queue.put(sentence_buffer.strip())
                await tts_queue.put(None); break
            print(token, end="", flush=True)
            sentence_buffer += token; full_response_text += token
            if any(term in token for term in sentence_terminators):
                await tts_queue.put(sentence_buffer.strip()); sentence_buffer = ""
        full_text_future.set_result(full_response_text)

    async def tts_consumer(self, tts_queue: asyncio.Queue):
        loop = asyncio.get_running_loop()
        while True:
            text_chunk = await tts_queue.get()
            if text_chunk is None: break
            audio_data = await loop.run_in_executor(self.services.tts_executor, self.tts_engine.synthesize_speech_sync, text_chunk)
            await self.services.player.add_to_queue(audio_data)

class ServiceManager:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop; self.audio_input_queue = queue.Queue(maxsize=100)
        self.raw_audio_queue = asyncio.Queue(maxsize=100); self.user_utterance_queue = asyncio.Queue()
        self.is_audio_running = threading.Event(); self.audio_input_thread = None
        self.vad_processor = VADProcessor(self.raw_audio_queue, self.user_utterance_queue)
        self.player = AudioPlayer()
        self.audio_poller_task = None
        self._create_executors()

    def _create_executors(self):
        logger.info("Creating dedicated thread pool executors for STT and TTS.")
        self.stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stt_worker')
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tts_worker')

    async def _audio_poller(self):
        logger.info("Audio poller task started.")
        loop = asyncio.get_running_loop()
        while self.is_audio_running.is_set():
            try:
                frame = await loop.run_in_executor(None, self.audio_input_queue.get)
                await self.raw_audio_queue.put(frame)
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"Error in audio poller: {e}"); break

    async def start(self):
        self.player.start()
        self.is_audio_running.set()
        self.audio_input_thread = threading.Thread(
            target=audio_input_thread_worker, args=(self.audio_input_queue, self.is_audio_running), daemon=True
        )
        self.audio_input_thread.start()
        self.audio_poller_task = asyncio.create_task(self._audio_poller())
        await self.vad_processor.start()

    async def stop(self):
        logger.info("Stopping agent services...")
        self.is_audio_running.clear()
        if self.audio_poller_task: self.audio_poller_task.cancel()
        if self.audio_input_thread: self.audio_input_thread.join(timeout=2)
        await self.vad_processor.stop()
        await self.player.stop()
        self._shutdown_executors()
        
    def _shutdown_executors(self):
        logger.info("Shutting down thread pool executors...")
        self.stt_executor.shutdown(wait=True); self.tts_executor.shutdown(wait=True)

    def reset_executors(self):
        logger.warning("Resetting STT and TTS executors...")
        self.stt_executor.shutdown(wait=False, cancel_futures=True)
        self.tts_executor.shutdown(wait=False, cancel_futures=True)
        self._create_executors()

class ConversationManager:
    def __init__(self, services: ServiceManager):
        self.state = AgentState.IDLE; self.services = services
        llm_instance = self._load_llm(); transcriber = Transcriber()
        llm_engine = LLMEngine(llm_instance); tts_engine = TextToSpeechEngine()
        self.pipeline_runner = PipelineRunner(transcriber, llm_engine, tts_engine, services)
        self.active_pipeline_task = None; self.current_agent_utterance = ""

    async def run(self):
        logger.info("Conversation Manager started. Listening for speech...")
        while True:
            user_audio_data = await self.services.user_utterance_queue.get()
            if self.state == AgentState.RESPONDING:
                is_barge_in = await self.check_for_barge_in(user_audio_data)
                if is_barge_in: await self.handle_barge_in(user_audio_data)
            elif self.state == AgentState.IDLE:
                if self.active_pipeline_task and not self.active_pipeline_task.done():
                    logger.warning("User spoke while processing. Ignoring.")
                    continue
                await self.start_new_turn(user_audio_data)

    async def start_new_turn(self, audio_data: np.ndarray):
        await self.transition_to(AgentState.PROCESSING)
        self.active_pipeline_task = asyncio.create_task(self._run_and_manage_pipeline(audio_data))

    async def _run_and_manage_pipeline(self, audio_data: np.ndarray):
        try:
            self.current_agent_utterance = await self.pipeline_runner.run(audio_data)
            await self.transition_to(AgentState.RESPONDING)
            logger.info("Data processing complete. Waiting for playback to finish...")
            while not self.services.player._playback_queue.empty(): await asyncio.sleep(0.1)
            while self.services.player._stream and self.services.player._stream.active: await asyncio.sleep(0.1)
        except asyncio.CancelledError: logger.warning("Pipeline task was cancelled by barge-in.")
        except Exception as e: logger.error(f"Error in pipeline: {e}", exc_info=True)
        finally:
            self.current_agent_utterance = ""; await self.transition_to(AgentState.IDLE)
            logger.info("Turn complete. Returning to IDLE.")

    async def check_for_barge_in(self, audio_data) -> bool:
        loop = asyncio.get_running_loop()
        user_text = await loop.run_in_executor(
            self.services.stt_executor, self.pipeline_runner.transcriber.transcribe_audio_sync, audio_data
        )
        if not user_text: return False
        if not self.is_echo(user_text):
            logger.info(f"Barge-in detected! User said: '{user_text}'"); return True
        return False

    def is_echo(self, user_text: str) -> bool:
        if not self.current_agent_utterance: return False
        agent_norm = ''.join(c for c in self.current_agent_utterance if c.isalnum()).lower()
        user_norm = ''.join(c for c in user_text if c.isalnum()).lower()
        if not user_norm: return True
        if user_norm in agent_norm:
            logger.debug(f"Echo detected: '{user_text}'"); return True
        return False

    async def handle_barge_in(self, audio_data: np.ndarray):
        logger.warning("Handling barge-in...")
        if self.active_pipeline_task: self.active_pipeline_task.cancel()
        await self.services.player.interrupt()
        self.services.reset_executors()
        await self.start_new_turn(audio_data)

    async def transition_to(self, new_state: AgentState):
        logger.info(f"State transition: {self.state.name} -> {new_state.name}")
        self.state = new_state
        
    def _load_llm(self):
        logger.info("Loading LLM... (This may take a moment)")
        try:
            from llama_cpp import Llama
            llm = Llama(model_path="./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=32768, verbose=False)
            logger.success("LLM loaded into memory.")
            return llm
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load LLM. Agent cannot start. Error: {e}"); return None

# --- MAIN EXECUTION BLOCK ---
async def main():
    loop = asyncio.get_running_loop()
    services = ServiceManager(loop)
    manager = ConversationManager(services)
    try:
        await services.start()
        await manager.run()
    finally:
        logger.info("Shutting down agent...")
        await services.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")