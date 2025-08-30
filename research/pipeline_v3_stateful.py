import asyncio
import threading
import sys
import time
import numpy as np
from scipy.io.wavfile import read
# from llama_cpp import Llama
import mlx_whisper
from enum import Enum, auto
from loguru import logger
import pyaudio
import webrtcvad
import collections
import torch
from concurrent.futures import ThreadPoolExecutor
# import soundfile as sf
import sounddevice as sd
import queue

# --- OBSERVABILITY SETUP ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/pipeline_v3_{time}.json", level="DEBUG", serialize=True, rotation="10 MB", catch=True)
logger.info("Logger configured. Starting Stateful Voice Agent.")

# --- VAD MODEL AND UTILITIES (LOADED ONCE AT STARTUP) ---
logger.info("Loading Silero VAD model from torch.hub...")
try:
    silero_model, utils_tuple = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = utils_tuple
    logger.success("Silero VAD model and utilities loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load Silero VAD model: {e}")
    sys.exit(1)

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
            logger.error(f"Transcription failed in worker thread: {e}", exc_info=True)
            return None
        elapsed_time = time.time() - start_time
        logger.success(f"Transcription complete in {elapsed_time:.2f}s. Text: '{transcribed_text}'")
        return transcribed_text

class TextToSpeechEngine:
    LOCAL_MODEL_PATH = "models/Kokoro"
    
    def __init__(self, speaker="af_heart"):
        self.SPEAKER = speaker
        logger.info(f"TTS worker initialized to use model path '{self.LOCAL_MODEL_PATH}'.")

    def synthesize_speech_sync(self, text_chunk: str, output_prefix: str) -> str | None:
        logger.info(f"Starting synchronous synthesis for chunk: '{text_chunk}'")
        start_time = time.time()
        try:
            from mlx_audio.tts.generate import generate_audio
            generate_audio(
                text=text_chunk, model_path=self.LOCAL_MODEL_PATH, voice=self.SPEAKER,
                file_prefix=output_prefix, verbose=False, play=False
            )
            import glob
            files = glob.glob(f"{output_prefix}*.wav")
            if files:
                audio_file = files[0]
                elapsed_time = time.time() - start_time
                logger.success(f"Synthesis complete in {elapsed_time:.2f}s, saved to '{audio_file}'")
                return audio_file
            else:
                logger.error(f"TTS ran but no output file was found for prefix '{output_prefix}'")
                return None
        except Exception as e:
            logger.error(f"Exception during synchronous synthesis: {e}", exc_info=True)
            return None

# ==================================================================================
# SECTION 2: BACKGROUND HARDWARE/SERVICE CLASSES
# ==================================================================================

def audio_input_thread_worker(
    audio_queue: queue.Queue,
    is_running_event: threading.Event,
    sample_rate=16000,
    frame_duration_ms=30
):
    logger.info("Audio input worker thread started.")
    p = pyaudio.PyAudio()
    chunk_size = int(sample_rate * frame_duration_ms / 1000)
    stream = None
    try:
        stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=sample_rate,
            input=True, frames_per_buffer=chunk_size
        )
        logger.success("PyAudio stream opened. Now reading frames.")
        while is_running_event.is_set():
            frame = stream.read(chunk_size, exception_on_overflow=False) # Don't crash on overflow
            try:
                # --- THE FIX: Use a non-blocking put with a timeout ---
                audio_queue.put(frame, block=False)
            except queue.Full:
                # This is our safety valve. If the app is too slow, we just drop a frame.
                logger.warning("Audio input queue is full, dropping a frame.")
                pass

    except Exception as e:
        logger.error(f"Error in audio input thread: {e}", exc_info=True)
    finally:
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()
        logger.info("PyAudio stream resources released.")


class VADProcessor:
    def __init__(self, raw_audio_queue: asyncio.Queue, utterance_queue: asyncio.Queue, sample_rate=16000, frame_duration_ms=30, padding_ms=300):
        self.raw_audio_queue = raw_audio_queue
        self.utterance_queue = utterance_queue
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.vad = webrtcvad.Vad(3)
        num_padding_frames = padding_ms // frame_duration_ms
        self.ring_buffer = collections.deque(maxlen=num_padding_frames)
        self.ratio = 0.75
        self.silero_model = silero_model
        self.vad_task = None

    def is_speech(self, frame: bytes) -> bool: return self.vad.is_speech(frame, self.sample_rate)

    async def start(self):
        logger.info("Starting Two-Stage VAD processor...")
        self.vad_task = asyncio.create_task(self._vad_collector())

    async def stop(self):
        if self.vad_task: 
            self.vad_task.cancel()
        await asyncio.sleep(0.1)
        logger.success("VAD processor stopped.")

    async def _vad_collector(self):
        triggered = False
        voiced_frames = []
        while True:
            try:
                frame = await self.raw_audio_queue.get()
                if not triggered:
                    self.ring_buffer.append((frame, self.is_speech(frame)))
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])
                    if num_voiced > self.ratio * self.ring_buffer.maxlen:
                        logger.debug("WebRTCVAD triggered.")
                        triggered = True
                        for f, s in self.ring_buffer: 
                            voiced_frames.append(f)
                        self.ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    self.ring_buffer.append((frame, self.is_speech(frame)))
                    num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                    if num_unvoiced > self.ratio * self.ring_buffer.maxlen:
                        logger.debug("WebRTCVAD un-triggered.")
                        triggered = False
                        utterance_bytes = b''.join(voiced_frames)
                        self.ring_buffer.clear()
                        voiced_frames = []
                        await self.confirm_with_silero(utterance_bytes)
            except asyncio.CancelledError: 
                break

    async def confirm_with_silero(self, audio_bytes: bytes):
        logger.info("WebRTCVAD detected utterance, confirming with Silero VAD...")
        audio_int16 = np.frombuffer(audio_bytes, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float32)
        loop = asyncio.get_running_loop()
        speech_timestamps = await loop.run_in_executor(
            None, lambda: get_speech_timestamps(audio_tensor, self.silero_model, sampling_rate=self.sample_rate)
        )
        if len(speech_timestamps) > 0:
            logger.success("Silero VAD confirmed speech. Emitting utterance.")
            await self.utterance_queue.put(audio_float32)
        else:
            logger.warning("Silero VAD detected NOISE. Discarding utterance.")

class AudioPlayer:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self._playback_queue = asyncio.Queue()
        self._playback_task = None
        self._stream = None
        logger.info("AudioPlayer initialized for persistent stream playback.")

    def start(self):
        try:
            self._stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='float32')
            self._stream.start()
            self._playback_task = asyncio.create_task(self._playback_loop())
            logger.success("AudioPlayer stream started and playback loop is running.")
        except Exception as e: 
            logger.critical(f"CRITICAL: Failed to open audio stream. Error: {e}")

    async def stop(self):
        await self._playback_queue.put(None)
        if self._playback_task:
            try: 
                await self._playback_task
            except asyncio.CancelledError: 
                pass
        if self._stream: 
            self._stream.stop()
            self._stream.close()
        logger.success("AudioPlayer stream stopped and closed.")

    async def add_to_queue(self, audio_filepath: str):
        if audio_filepath: 
            await self._playback_queue.put(audio_filepath)

    async def interrupt(self):
        logger.warning("AudioPlayer received interrupt signal!")
        while not self._playback_queue.empty():
            try: 
                self._playback_queue.get_nowait()
            except asyncio.QueueEmpty: 
                break
        logger.info("Playback queue cleared.")
        sd.stop() 
        logger.info("Audio stream stopped immediately.")

    async def _playback_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                audio_filepath = await self._playback_queue.get()
                if audio_filepath is None: 
                    break
                logger.info(f"Player loop processing: '{audio_filepath}'")
                sample_rate, audio_data = await loop.run_in_executor(None, read, audio_filepath)
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                if self._stream:
                    await loop.run_in_executor(None, self._stream.write, audio_data)
                logger.success(f"Player loop finished writing: '{audio_filepath}'")
            except asyncio.CancelledError: 
                break
            except Exception as e: 
                logger.error(f"Error in playback loop: {e}", exc_info=True)

# ==================================================================================
# SECTION 3: ASYNC ORCHESTRATION CLASSES
# ==================================================================================

class AgentState(Enum):
    IDLE = auto()
    PROCESSING = auto()
    SPEAKING = auto()

class LLMEngine:
    def __init__(self, llm_instance):
        self.llm = llm_instance
        logger.success("LLMEngine initialized with pre-loaded model.")

    async def generate_response(self, user_prompt: str, text_queue: asyncio.Queue):
        prompt_messages = [
            {"role": "system", "content": "You are a helpful, brief, and conversational AI assistant. Your responses must be broken into complete sentences, ending with proper punctuation. Do not use emojis."},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(f"Starting LLM stream for prompt: '{user_prompt}'")
        start_time = time.time()
        loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(
            None, lambda: self.llm.create_chat_completion(messages=prompt_messages, max_tokens=150, stream=True)
        )
        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                token = delta['content']
                await text_queue.put(token)
        await text_queue.put(None)
        logger.success(f"LLM stream finished in {time.time() - start_time:.2f}s.")

class PipelineRunner:
    def __init__(self, transcriber: Transcriber, llm_engine: LLMEngine, tts_engine: TextToSpeechEngine, services: 'ServiceManager'):
        self.transcriber = transcriber 
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine
        self.services = services

    async def run(self, audio_data: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        transcribed_text = await loop.run_in_executor(
            self.services.stt_executor, self.transcriber.transcribe_audio_sync, audio_data
        )
        if not transcribed_text: 
            raise ValueError("Transcription failed.")
        
        text_token_queue = asyncio.Queue()
        tts_sentence_queue = asyncio.Queue()
        full_text_future = asyncio.Future()
        
        llm_task = asyncio.create_task(
            self.llm_engine.generate_response(transcribed_text, text_token_queue)
        )
        chunker_task = asyncio.create_task(
            self.text_chunker(text_token_queue, tts_sentence_queue, full_text_future)
        )
        tts_task = asyncio.create_task(
            self.tts_consumer(tts_sentence_queue, timestamp)
        )
        
        await asyncio.gather(llm_task, chunker_task, tts_task)
        return await full_text_future

    async def text_chunker(self, text_queue: asyncio.Queue, tts_queue: asyncio.Queue, full_text_future: asyncio.Future):
        sentence_buffer = ""
        full_response_text = ""
        sentence_terminators = {".", "?", "!"}
        while True:
            token = await text_queue.get()
            if token is None:
                if sentence_buffer.strip(): 
                    await tts_queue.put(sentence_buffer.strip())
                await tts_queue.put(None)
                break
            print(token, end="", flush=True)
            sentence_buffer += token
            full_response_text += token
            if any(term in token for term in sentence_terminators):
                await tts_queue.put(sentence_buffer.strip())
                sentence_buffer = ""
        full_text_future.set_result(full_response_text)

    async def tts_consumer(self, tts_queue: asyncio.Queue, timestamp: str):
        loop = asyncio.get_running_loop()
        chunk_index = 0
        while True:
            text_chunk = await tts_queue.get()
            if text_chunk is None: 
                break
            output_prefix = f"tts_output_{timestamp}_{chunk_index:02d}"
            audio_file = await loop.run_in_executor(
                self.services.tts_executor, self.tts_engine.synthesize_speech_sync, text_chunk, output_prefix
            )
            await self.services.player.add_to_queue(audio_file)
            chunk_index += 1

class ServiceManager:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.audio_input_queue = queue.Queue(maxsize=100) # Bounded size
        self.raw_audio_queue = asyncio.Queue(maxsize=100) # Bounded size
        self.user_utterance_queue = asyncio.Queue()
        self.is_audio_running = threading.Event() 
        self.audio_input_thread = None
        self.vad_processor = VADProcessor(self.raw_audio_queue, self.user_utterance_queue)
        self.player = AudioPlayer()
        self.audio_poller_task = None
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
            except asyncio.CancelledError: 
                break
            except Exception as e: 
                logger.error(f"Error in audio poller: {e}")
                break

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
        await self.player.stop()
        self.is_audio_running.clear()
        if self.audio_poller_task: 
            self.audio_poller_task.cancel()
        if self.audio_input_thread: 
            self.audio_input_thread.join(timeout=2)
        await self.vad_processor.stop()
        logger.info("Shutting down thread pool executors...")
        self.stt_executor.shutdown(wait=True)
        self.tts_executor.shutdown(wait=True)

class ConversationManager:
    def __init__(self, services: ServiceManager):
        self.state = AgentState.IDLE
        self.services = services
        llm_instance = self._load_llm() 
        transcriber = Transcriber()
        llm_engine = LLMEngine(llm_instance)
        tts_engine = TextToSpeechEngine()
        self.pipeline_runner = PipelineRunner(transcriber, llm_engine, tts_engine, services)
        self.active_pipeline_task = None
        self.current_agent_utterance = ""

    async def run(self):
        logger.info("Conversation Manager started. Listening for speech...")
        while True:
            user_audio_data = await self.services.user_utterance_queue.get()
            if self.state == AgentState.SPEAKING:
                is_barge_in = await self.check_for_barge_in(user_audio_data)
                if is_barge_in: 
                    await self.handle_barge_in(user_audio_data)
            elif self.state in [AgentState.IDLE, AgentState.PROCESSING]:
                if self.active_pipeline_task and not self.active_pipeline_task.done():
                    logger.warning("User spoke while processing previous turn. Ignoring for now.")
                    continue
                await self.start_new_turn(user_audio_data)

    async def start_new_turn(self, audio_data: np.ndarray):
        await self.transition_to(AgentState.PROCESSING)
        self.active_pipeline_task = asyncio.create_task(self._run_and_manage_pipeline(audio_data))

    async def _run_and_manage_pipeline(self, audio_data: np.ndarray):
        try:
            await self.transition_to(AgentState.SPEAKING)
            self.current_agent_utterance = await self.pipeline_runner.run(audio_data)
            logger.info("Data processing complete. Waiting for playback to finish...")
            while not self.services.player._playback_queue.empty(): 
                await asyncio.sleep(0.1)
            # A small extra wait to catch the tail end of the last playback
            while self.services.player._stream and self.services.player._stream.active: 
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
             logger.warning("Pipeline task was cancelled by barge-in.")
        except Exception as e: 
            logger.error(f"Error in pipeline: {e}", exc_info=True)
        finally:
            self.current_agent_utterance = ""
            await self.transition_to(AgentState.IDLE)
            logger.info("Turn complete. Returning to IDLE.")

    async def check_for_barge_in(self, audio_data) -> bool:
        loop = asyncio.get_running_loop()
        user_text = await loop.run_in_executor(
            self.services.stt_executor, self.pipeline_runner.transcriber.transcribe_audio_sync, audio_data
        )
        if not user_text: 
            return False
        if not self.is_echo(user_text):
            logger.info(f"Barge-in detected! User said: '{user_text}'")
            return True
        return False

    def is_echo(self, user_text: str) -> bool:
        if not self.current_agent_utterance:
             return False
        agent_norm = ''.join(c for c in self.current_agent_utterance if c.isalnum()).lower()
        user_norm = ''.join(c for c in user_text if c.isalnum()).lower()
        if not user_norm:
             return True
        if user_norm in agent_norm:
            logger.debug(f"Echo detected: '{user_text}'") 
            return True
        return False

    async def handle_barge_in(self, audio_data: np.ndarray):
        logger.warning("Handling barge-in...")
        if self.active_pipeline_task: 
            self.active_pipeline_task.cancel()
        await self.services.player.interrupt()
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
            logger.critical(f"CRITICAL: Failed to load LLM. Agent cannot start. Error: {e}")
            return None

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
# #pipeline_v3_stateful.py
# from concurrent.futures import ThreadPoolExecutor
# import asyncio
# import threading
# import sys
# import time
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import read #,write
# # from llama_cpp import Llama
# import mlx_whisper
# from enum import Enum, auto
# from loguru import logger
# from mlx_audio.tts.generate import generate_audio
# # from pynput import keyboard
# import queue 
# import pyaudio
# import webrtcvad
# import collections
# import torch
# # ---OBSERVABILITY ---
# # Remove the default handler to prevent duplicate outputs
# logger.remove()

# # Configure console logger
# # This will print color-coded logs to your terminal.
# logger.add(
#     sys.stderr, 
#     level="INFO",
#     format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
# )

# # Configure file logger
# # This will write structured JSON logs to a file, perfect for later analysis.
# logger.add(
#     "logs/pipeline_v1_{time}.json", 
#     level="DEBUG",
#     serialize=True, # This is the key for structured logging!
#     rotation="10 MB", # Rotates the log file when it reaches 10 MB
#     catch=True # Catches exceptions to prevent crashes in logging
# )

# logger.info("Logger configured. Starting the Traceable Pipe v3.")


# logger.info("Loading Silero VAD model from torch.hub...")
# try:
#     # trust_repo=True is needed to suppress the warning and run non-interactively
#     # We unpack the main tuple of (model, utils_tuple)
#     silero_model, utils_tuple = torch.hub.load(
#         repo_or_dir='snakers4/silero-vad',
#         model='silero_vad',
#         force_reload=False,
#         trust_repo=True
#     )
    
#     # Now, we unpack the nested tuple of utility functions
#     (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils_tuple
#     logger.success("Silero VAD model and utilities loaded successfully.")

# except Exception as e:
#     logger.critical(f"Failed to load Silero VAD model: {e}")
#     # Exit if VAD can't be loaded, as it's critical
#     sys.exit(1)

# class AudioRecorder:
#     def __init__(self, sample_rate=16000):
#         self.sample_rate = sample_rate
#         self.stream = None
#         self.audio_queue = queue.Queue() # A thread-safe queue to pass audio chunks

#     def _audio_callback(self, indata, frames, time, status):
#         """This is called by the sounddevice stream for each new audio chunk."""
#         self.audio_queue.put(indata.copy())

#     def start_recording(self):
#         """Starts the non-blocking audio stream."""
#         if self.stream is not None:
#             self.stop_recording() # Ensure any old stream is closed

#         logger.info("Starting audio stream...")
#         self.stream = sd.InputStream(
#             samplerate=self.sample_rate,
#             channels=1,
#             dtype='float32',
#             callback=self._audio_callback
#         )
#         self.stream.start()

#     def stop_recording(self) -> np.ndarray:
#         """Stops the stream and returns the full recorded audio."""
#         if self.stream is None:
#             return np.array([], dtype=np.float32)

#         logger.info("Stopping audio stream...")
#         self.stream.stop()
#         self.stream.close()
#         self.stream = None

#         # Drain the queue and concatenate all chunks
#         audio_chunks = []
#         while not self.audio_queue.empty():
#             audio_chunks.append(self.audio_queue.get())
        
#         if not audio_chunks:
#             return np.array([], dtype=np.float32)
            
#         return np.concatenate(audio_chunks, axis=0)


# # --- 1.2: SPEECH-TO-TEXT SUB-SYSTEM ---
# class Transcriber:
#     """
#     A synchronous worker class for transcribing audio.
#     Its methods are designed to be run in a separate thread.
#     """
#     def __init__(self, model_path="models/whisper-large-v3-turbo"):
#         self.model_path = model_path
#         logger.info(f"Transcriber worker initialized with model: '{self.model_path}'.")

#     def transcribe_audio_sync(self, audio_data: np.ndarray) -> str | None:
#         """
#         A simple, blocking function that transcribes an audio waveform.
#         """
#         logger.info("Starting synchronous transcription...")
#         start_time = time.time()
        
#         try:
#             # mlx-whisper can take the NumPy array directly
#             result = mlx_whisper.transcribe(
#                 audio=audio_data,
#                 path_or_hf_repo=self.model_path,
#                 language="en"
#             )
#             transcribed_text = result["text"].strip() if result else None
            
#         except Exception as e:
#             logger.error(f"Transcription failed in worker thread: {e}")
#             return None

#         elapsed_time = time.time() - start_time
#         logger.success(f"Transcription complete in {elapsed_time:.2f}s.")
#         logger.info(f"Transcribed Text: '{transcribed_text}'")
        
#         return transcribed_text


# # --- 1.3: LLM COGNITIVE SUB-SYSTEM ---
# class LLMEngine:
#     def __init__(self, llm_instance):
#         if llm_instance is None:
#             raise ValueError("LLM instance cannot be None.")
#         self.llm = llm_instance
#         logger.success("LLMEngine initialized with pre-loaded model.")

#     async def generate_response(self, user_prompt: str, text_queue: asyncio.Queue):
#         # --- THIS METHOD IS NOW A GENERATOR ---
        
#         prompt_messages = [
#             {
#                 "role": "system",
#                 "content": "You are a helpful, brief, and conversational AI assistant. Your responses must be broken into complete sentences, ending with proper punctuation with no emojis. This is critical for the text-to-speech system.",
#             },
#             {"role": "user", "content": user_prompt},
#         ]
        
#         logger.info(f"Starting LLM stream for prompt: '{user_prompt}'")
#         # --- TRACEPOINT START ---
#         start_time = time.time()

#         # The blocking call will be run in a separate thread.
#         loop = asyncio.get_running_loop()
#         stream = await loop.run_in_executor(
#             None,  # Use the default thread pool executor
#             lambda: self.llm.create_chat_completion(
#                 messages=prompt_messages, max_tokens=150, stream=True
#             )
#         )
        
#         full_response_text = ""
#         # Iterate over the stream of chunks
#         for chunk in stream:
#             # Extract the token from the chunk
#             delta = chunk['choices'][0]['delta']
#             if 'content' in delta:
#                 token = delta['content']
#                 full_response_text += token
#                 await text_queue.put(token) # Put token into the async queue
       
#         await text_queue.put(None) # Sentinel value to signal the end of the stream
#         logger.success(f"LLM stream finished in {time.time() - start_time:.2f}s.")

        


# class TextToSpeechEngine:
#     """
#     A synchronous worker class for synthesizing speech.
#     Its methods are designed to be run in a separate thread.
#     """
#     LOCAL_MODEL_PATH = "models/Kokoro"
    
#     def __init__(self, speaker="af_heart"):
#         self.SPEAKER = speaker
#         logger.info(f"TTS worker initialized to use model path '{self.LOCAL_MODEL_PATH}'.")

#     def synthesize_speech_sync(self, text_chunk: str, output_prefix: str) -> str | None:
#         """
#         A simple, blocking function that synthesizes text and saves it to a file.
#         """
#         logger.info(f"Starting synchronous synthesis for chunk: '{text_chunk}'")
#         start_time = time.time()
        
#         try:
#             # This is our known-good, high-level function call
#             generate_audio(
#                 text=text_chunk,
#                 model_path=self.LOCAL_MODEL_PATH,
#                 voice=self.SPEAKER,
#                 file_prefix=output_prefix,
#                 verbose=False,
#                 play=False
#             )

#             # Find the generated file
#             import glob
#             files = glob.glob(f"{output_prefix}*.wav")
#             if files:
#                 audio_file = files[0]
#                 elapsed_time = time.time() - start_time
#                 logger.success(f"Synthesis complete in {elapsed_time:.2f}s, saved to '{audio_file}'")
#                 return audio_file
#             else:
#                 logger.error(f"TTS ran but no output file was found for prefix '{output_prefix}'")
#                 return None
#         except Exception as e:
#             logger.error(f"Exception during synchronous synthesis: {e}")
#             return None


# class AudioPlayer:
#     def __init__(self, sample_rate=24000): # Kokoro's default sample rate
#         self.sample_rate = sample_rate
#         self._playback_queue = asyncio.Queue()
#         self._playback_task = None
#         self._stream = None # This will hold our persistent audio stream
#         self._stop_event = threading.Event()
#         logger.info("AudioPlayer initialized for persistent stream playback.")

#     def start(self):
#         """Initializes the audio stream and starts the playback loop."""
#         # --- THE FIX: Create ONE stream that lives forever ---
#         try:
#             self._stream = sd.OutputStream(
#                 samplerate=self.sample_rate,
#                 channels=1,
#                 dtype='float32'
#             )
#             self._stream.start()
#             self._playback_task = asyncio.create_task(self._playback_loop())
#             logger.success("AudioPlayer stream started and playback loop is running.")
#         except Exception as e:
#             logger.critical(f"CRITICAL: Failed to open audio stream. Error: {e}")

#     async def stop(self):
#         """Stops the playback loop and closes the audio stream gracefully."""
#         await self._playback_queue.put(None)
#         if self._playback_task:
#             await self._playback_task
        
#         if self._stream:
#             self._stream.stop()
#             self._stream.close()
#             logger.success("AudioPlayer stream stopped and closed.")

#     async def add_to_queue(self, audio_filepath: str):
#         if audio_filepath:
#             await self._playback_queue.put(audio_filepath)

    
#     async def interrupt(self):
#         logger.warning("AudioPlayer received interrupt signal!")
#         while not self._playback_queue.empty():
#             try:
#                  self._playback_queue.get_nowait()
#             except asyncio.QueueEmpty:
#                  break
#         logger.info("Playback queue cleared.")
        
#         # sd.stop() is thread-safe and will interrupt sd.wait()
#         sd.stop()
#         logger.info("Audio stream stopped immediately.")

#     def _blocking_play(self, audio_filepath):
#         try:
#             sample_rate, audio_data = read(audio_filepath)
#             sd.play(audio_data, sample_rate)
#             sd.wait() # This will be interrupted by sd.stop()
#         except Exception as e:
#             logger.error(f"Playback failed for {audio_filepath}: {e}")
    

#     async def _playback_loop(self):
#         """The core loop that reads files, CONVERTS DTYPE, and WRITES to the persistent stream."""
#         loop = asyncio.get_running_loop()
        
#         while True:
#             audio_filepath = await self._playback_queue.get()
#             if audio_filepath is None:
#                 break
            
#             logger.info(f"Player loop processing: '{audio_filepath}'")
            
#             try:
#                 # Read the audio file (likely as int16)
#                 sample_rate, audio_data = await loop.run_in_executor(
#                     None, read, audio_filepath
#                 )
#                 if sample_rate != self.sample_rate:
#                     logger.warning(f"Audio file sample rate ({sample_rate}) differs from stream rate ({self.sample_rate}).")
                
#                 # --- THE FIX: DTYPE CONVERSION AND NORMALIZATION ---
#                 # Check if the data is int16
#                 if audio_data.dtype == np.int16:
#                     # Convert to float32 and normalize from [-32768, 32767] to [-1.0, 1.0]
#                     audio_data = audio_data.astype(np.float32) / 32768.0
#                     logger.debug("Converted audio data from int16 to float32 for playback.")
#                 # --- END OF FIX ---

#             except Exception as e:
#                 logger.error(f"Failed to read or convert audio file {audio_filepath}: {e}")
#                 continue

#             if self._stream:
#                 try:
#                     await loop.run_in_executor(
#                         None, self._stream.write, audio_data
#                     )
#                     logger.success(f"Player loop finished writing: '{audio_filepath}'")
#                 except Exception as e:
#                     logger.error(f"Error writing to audio stream: {e}")



# # --- STATES AND EVENTS ---
# class AgentState(Enum):
#     IDLE = auto()
#     LISTENING = auto()
#     PROCESSING = auto()
#     SPEAKING = auto()

# class ServiceManager:
#     def __init__(self, loop: asyncio.AbstractEventLoop):
#         self.loop = loop

#         self.raw_audio_queue = asyncio.Queue()
#         self.user_utterance_queue = asyncio.Queue()
        
#         self.audio_streamer = AudioStreamer(self.raw_audio_queue, self.loop)
#         self.vad_processor = VADProcessor(self.raw_audio_queue, self.user_utterance_queue)
#         self.player = AudioPlayer()

#         logger.info("Creating dedicated thread pool executors for STT and TTS.")
#         self.stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stt_worker')
#         self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tts_worker')


#     async def start(self):
#         self.player.start()
#         self.audio_streamer.start()
#         await self.vad_processor.start()

#     async def stop(self):
#         await self.player.stop()
#         self.audio_streamer.stop()
#         await self.vad_processor.stop()
#         logger.info("Shutting down thread pool executors...")
#         self.stt_executor.shutdown(wait=True)
#         self.tts_executor.shutdown(wait=True)


# # --- THE  STATEFUL VOICE AGENT ---

# class PipelineRunner:
#     def __init__(self, transcriber, llm_engine, tts_engine, services: ServiceManager): #player,
#         self.transcriber = transcriber
#         self.llm_engine = llm_engine
#         self.tts_engine = tts_engine
#         # self.player = player
#         self.services = services
#         self.full_response_text = ""
        

#     async def run(self, audio_data: np.ndarray):
#         loop = asyncio.get_running_loop()
#         timestamp = time.strftime("%Y%m%d-%H%M%S")

#         # 1. Schedule transcription on the STT executor
#         transcribed_text = await loop.run_in_executor(
#             self.services.stt_executor,
#             self.transcriber.transcribe_audio_sync, # Call the new sync method
#             audio_data
#         )
#         if not transcribed_text:
#              raise ValueError("Transcription failed.")
        
#         # 2. Setup streaming pipeline
#         text_token_queue = asyncio.Queue()
#         tts_sentence_queue = asyncio.Queue()
#         full_text_future = asyncio.Future()
        
#         # 3. Start concurrent tasks
#         llm_task = asyncio.create_task(
#             self.llm_engine.generate_response(transcribed_text, text_token_queue)
#         )
#         chunker_task = asyncio.create_task(
#             self.text_chunker(text_token_queue, tts_sentence_queue, full_text_future) # <-- PASS THE FUTURE
#         )
#         tts_task = asyncio.create_task(
#             self.tts_consumer(tts_sentence_queue, timestamp)
#         )
        
#         await asyncio.gather(llm_task, chunker_task, tts_task)
        
#         # 4. Return the completed full text
#         return await full_text_future

#     def _transcribe_sync(self, audio_data: np.ndarray) -> str | None:
#         result = mlx_whisper.transcribe(
#             audio=audio_data,
#             path_or_hf_repo=self.transcriber.model_path,
#             language="en"
#         )
#         return result["text"].strip() if result else None

#     async def text_chunker(self, text_queue: asyncio.Queue, tts_queue: asyncio.Queue, full_text_future: asyncio.Future):
#         sentence_buffer = ""
#         full_response_text = ""
#         sentence_terminators = {".", "?", "!"}
        
#         while True:
#             token = await text_queue.get()
#             if token is None:
#                 if sentence_buffer.strip():
#                     await tts_queue.put(sentence_buffer.strip())
#                 await tts_queue.put(None)
#                 break
            
#             print(token, end="", flush=True)
#             sentence_buffer += token
#             full_response_text += token # <-- CAPTURE THE FULL TEXT
#             if any(term in token for term in sentence_terminators):
#                 await tts_queue.put(sentence_buffer.strip())
#                 sentence_buffer = ""
        
#         full_text_future.set_result(full_response_text)

    
#     async def tts_consumer(self, tts_queue: asyncio.Queue, timestamp: str):
#         loop = asyncio.get_running_loop()
#         chunk_index = 0
#         while True:
#             text_chunk = await tts_queue.get()
#             if text_chunk is None:
#                  break
            
#             output_prefix = f"tts_output_{timestamp}_{chunk_index:02d}"
            
#             # Schedule synthesis on the TTS executor
#             audio_file = await loop.run_in_executor(
#                 self.services.tts_executor,
#                 self.tts_engine.synthesize_speech_sync, # Call the new sync method
#                 text_chunk,
#                 output_prefix
#             )
#             await self.services.player.add_to_queue(audio_file)
#             chunk_index += 1


# # --- THE CLEANED UP ConversationManager (formerly VoiceAgent) ---
# class ConversationManager:
#     def __init__(self, services: ServiceManager):
#         self.state = AgentState.IDLE
#         self.services = services
        
#         # Load models and create the pipeline worker
#         llm_instance = self._load_llm()
#         transcriber = Transcriber()
#         llm_engine = LLMEngine(llm_instance)
#         tts_engine = TextToSpeechEngine()
#         # self.pipeline_runner = PipelineRunner(transcriber, llm_engine, tts_engine, services)
#         self.pipeline_runner = PipelineRunner(transcriber, llm_engine, tts_engine, services)
        

        
#         self.active_pipeline_task = None
#         self.current_agent_utterance = ""

#     def is_echo(self, user_text: str) -> bool:
#         """A simple software-based echo cancellation check."""
#         if not self.current_agent_utterance:
#             return False
            
#         # Normalize both strings
#         agent_norm = ''.join(c for c in self.current_agent_utterance if c.isalnum()).lower()
#         user_norm = ''.join(c for c in user_text if c.isalnum()).lower()

#         if not user_norm: # Ignore empty/silent transcriptions
#             return True 

#         # Check if the user's speech is a substring of the agent's speech
#         if user_norm in agent_norm:
#             logger.debug(f"Echo detected. User said '{user_text}', which is a substring of agent's speech.")
#             return True
            
#         return False

#     async def run(self):
#         logger.info("Conversation Manager started. Listening for speech...")
#         while True:
#             user_audio_data = await self.services.user_utterance_queue.get()

#             if self.state == AgentState.SPEAKING:
#                 is_barge_in = await self.check_for_barge_in(user_audio_data)
#                 if is_barge_in:
#                     await self.handle_barge_in(user_audio_data)
#             elif self.state == AgentState.IDLE or self.state == AgentState.PROCESSING:
#                 if self.active_pipeline_task and not self.active_pipeline_task.done():
#                     logger.warning("User spoke while processing previous turn. Ignoring for now.")
#                     continue
                
#                 await self.start_new_turn(user_audio_data)

#     async def start_new_turn(self, audio_data: np.ndarray):
#         await self.transition_to(AgentState.PROCESSING)
#         self.active_pipeline_task = asyncio.create_task(self._run_and_manage_pipeline(audio_data))

#     async def _run_and_manage_pipeline(self, audio_data: np.ndarray):
#         try:
#             await self.transition_to(AgentState.SPEAKING)
#             self.current_agent_utterance = await self.pipeline_runner.run(audio_data)
            
#             # Wait for playback to finish
#             while not self.services.player._playback_queue.empty():
#                 await asyncio.sleep(0.1)
#         except asyncio.CancelledError:
#             logger.warning("Pipeline task was cancelled due to barge-in.")
#         except Exception as e:
#             logger.error(f"Error in pipeline: {e}")
#         finally:
#             self.current_agent_utterance = ""
#             await self.transition_to(AgentState.IDLE)
#             logger.info("Turn complete. Returning to IDLE.")
            
#     async def check_for_barge_in(self, audio_data) -> bool:
#         loop = asyncio.get_running_loop()
        
#         # Schedule the barge-in transcription on the STT executor
#         user_text = await loop.run_in_executor(
#             self.services.stt_executor,
#             self.pipeline_runner.transcriber.transcribe_audio_sync, # Call the new sync method
#             audio_data
#         )
        
#         if not user_text:
#              return False
        
#         # Echo cancellation logic (is_echo function needs to be a method of this class)
#         if not self.is_echo(user_text):
#             logger.info(f"Barge-in detected! User said: '{user_text}'")
#             return True
#         else:
#             logger.debug(f"Echo detected and ignored: '{user_text}'")
#             return False

#     async def handle_barge_in(self, audio_data: np.ndarray):
#         logger.warning("Handling barge-in...")
#         if self.active_pipeline_task:
#             self.active_pipeline_task.cancel()
#         await self.services.player.interrupt()
#         await self.start_new_turn(audio_data)

#     async def transition_to(self, new_state: AgentState):
#         logger.info(f"State transition: {self.state.name} -> {new_state.name}")
#         self.state = new_state

#     def _load_llm(self):
#         # This private method centralizes the heavy LLM loading.
#         logger.info("Loading LLM... (This may take a moment)")
#         try:
#             from llama_cpp import Llama
#             llm = Llama(
#                 model_path="./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
#                 n_gpu_layers=-1,
#                 n_ctx=32768,
#                 verbose=False
#             )
#             logger.success("LLM loaded into memory.")
#             return llm
#         except Exception as e:
#             logger.critical(f"CRITICAL: Failed to load LLM. Agent cannot start. Error: {e}")
#             return None

# class AudioStreamer:
#     """Uses PyAudio to continuously read audio in a dedicated background thread."""
#     def __init__(self, raw_audio_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, sample_rate=16000, frame_duration_ms=30):
#         self.raw_audio_queue = raw_audio_queue
#         self.loop = loop
#         self.sample_rate = sample_rate
#         self.frame_duration_ms = frame_duration_ms
#         self.chunk_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
#         self.p = pyaudio.PyAudio()
#         self.stream = None
#         self.is_running = threading.Event()

#     def start(self):
#         """Starts the audio streaming in a dedicated background thread."""
#         logger.info("Starting PyAudio streamer...")
#         self.is_running.set()
#         thread = threading.Thread(target=self._run_stream, daemon=True)
#         thread.start()
#         logger.success("PyAudio streamer thread started.")

#     def _run_stream(self):
#         """This function runs in its own dedicated thread."""
#         self.stream = self.p.open(
#             format=pyaudio.paInt16,
#             channels=1,
#             rate=self.sample_rate,
#             input=True,
#             frames_per_buffer=self.chunk_size
#         )
#         logger.info("PyAudio stream opened. Now reading frames...")
        
#         # --- THE CORE FIX: A robust, blocking read loop ---
#         while self.is_running.is_set():
#             try:
#                 # Read audio data from the stream. This is a blocking call.
#                 frame = self.stream.read(self.chunk_size)
#                 # Safely put the data onto the asyncio queue from this thread
#                 self.loop.call_soon_threadsafe(self.raw_audio_queue.put_nowait, frame)
#             except IOError as e:
#                 # This can happen if the stream is closed, e.g., on shutdown
#                 logger.warning(f"PyAudio IOError: {e}")
#                 break
#             except Exception as e:
#                 logger.error(f"Error in audio streaming loop: {e}")
#                 break

#         # Cleanup
#         self.stream.stop_stream()
#         self.stream.close()
#         self.p.terminate()
#         logger.info("PyAudio stream resources released.")

#     def stop(self):
#         """Signals the streaming thread to stop."""
#         logger.info("Stopping PyAudio streamer...")
#         self.is_running.clear()

# class VADProcessor:
#     """
#     A two-stage VAD that uses a lightweight WebRTCVAD for initial filtering
#     and a more powerful Silero VAD for final confirmation.
#     """
#     def __init__(self, raw_audio_queue: asyncio.Queue, utterance_queue: asyncio.Queue, sample_rate=16000, frame_duration_ms=30, padding_ms=300):
#         self.raw_audio_queue = raw_audio_queue
#         self.utterance_queue = utterance_queue
#         self.sample_rate = sample_rate
#         self.frame_duration_ms = frame_duration_ms
        
#         # WebRTCVAD setup
#         self.vad = webrtcvad.Vad(3) # Aggressiveness level 3
        
#         # Ring buffer logic from the example
#         num_padding_frames = padding_ms // frame_duration_ms
#         self.ring_buffer = collections.deque(maxlen=num_padding_frames)
#         self.ratio = 0.75 # Ratio of speech frames to trigger
        
#         # Silero VAD setup
#         self.silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        
#         self.vad_task = None

#     def is_speech(self, frame: bytes) -> bool:
#         """Uses WebRTCVAD to check a single frame for speech."""
#         return self.vad.is_speech(frame, self.sample_rate)

#     async def start(self):
#         logger.info("Starting Two-Stage VAD processor...")
#         self.vad_task = asyncio.create_task(self._vad_collector())

#     async def stop(self):
#         logger.info("Stopping VAD processor...")
#         if self.vad_task:
#             self.vad_task.cancel()
#             try:
#                  await self.vad_task
#             except asyncio.CancelledError:
#                  pass
#         logger.success("VAD processor stopped.")

#     async def _vad_collector(self):
#         """
#         Consumes raw audio and collects utterances based on WebRTCVAD,
#         then confirms them with Silero VAD.
#         """
#         triggered = False
#         voiced_frames = []
        
#         while True:
#             try:
#                 frame = await self.raw_audio_queue.get()

#                 if len(frame) != int(self.sample_rate * self.frame_duration_ms / 1000 * 2): # 2 bytes per sample
#                     continue
                
#                 is_speech_frame = self.is_speech(frame)

#                 if not triggered:
#                     self.ring_buffer.append((frame, is_speech_frame))
#                     num_voiced = len([f for f, speech in self.ring_buffer if speech])
#                     if num_voiced > self.ratio * self.ring_buffer.maxlen:
#                         logger.debug("WebRTCVAD triggered.")
#                         triggered = True
#                         for f, s in self.ring_buffer:
#                             voiced_frames.append(f)
#                         self.ring_buffer.clear()
#                 else:
#                     voiced_frames.append(frame)
#                     self.ring_buffer.append((frame, is_speech_frame))
#                     num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
#                     if num_unvoiced > self.ratio * self.ring_buffer.maxlen:
#                         logger.debug("WebRTCVAD un-triggered.")
#                         triggered = False
                        
#                         # --- Utterance complete, now confirm with Silero ---
#                         utterance_bytes = b''.join(voiced_frames)
#                         self.ring_buffer.clear()
#                         voiced_frames = []
                        
#                         await self.confirm_with_silero(utterance_bytes)
#             except asyncio.CancelledError:
#                 break

#     async def confirm_with_silero(self, audio_bytes: bytes):
#         """Runs the collected utterance through Silero VAD for final confirmation."""
#         logger.info("WebRTCVAD detected utterance, confirming with Silero VAD...")
        
#         # Convert bytes to float32 tensor for Silero
#         audio_int16 = np.frombuffer(audio_bytes, np.int16)
#         audio_float32 = audio_int16.astype(np.float32) / 32768.0
#         audio_tensor = torch.from_numpy(audio_float32)

#         # Run Silero VAD
#         loop = asyncio.get_running_loop()
#         speech_timestamps = await loop.run_in_executor(
#             None, lambda: get_speech_timestamps(audio_tensor, self.silero_model, sampling_rate=self.sample_rate) 
#         )

#         if len(speech_timestamps) > 0:
#             logger.success("Silero VAD confirmed speech. Emitting utterance.")
#             # We already have the float32 data, so we can put it directly
#             await self.utterance_queue.put(audio_float32)
#         else:
#             logger.warning("Silero VAD detected NOISE. Discarding utterance.")

# # --- MAIN EXECUTION BLOCK ---
# async def main():
#     loop = asyncio.get_running_loop()
#     services = ServiceManager(loop)
#     manager = ConversationManager(services)
    
#     try:
#         await services.start()
#         await manager.run()
#     finally:
#         logger.info("Shutting down agent...")
#         await services.stop()

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         logger.info("Shutdown requested.")
    
   