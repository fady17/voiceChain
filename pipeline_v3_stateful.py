#pipeline_v3_stateful.py
import asyncio
import threading
import sys
import time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import read #,write
# from llama_cpp import Llama
import mlx_whisper
from enum import Enum, auto
from loguru import logger
from mlx_audio.tts.generate import generate_audio
# from pynput import keyboard
import queue 
import pyaudio
import webrtcvad
import collections
import torch
# --- 1.0: CONFIGURE OBSERVABILITY ---
# Remove the default handler to prevent duplicate outputs
logger.remove()

# Configure console logger
# This will print color-coded logs to your terminal.
logger.add(
    sys.stderr, 
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Configure file logger
# This will write structured JSON logs to a file, perfect for later analysis.
logger.add(
    "logs/pipeline_v1_{time}.json", 
    level="DEBUG",
    serialize=True, # This is the key for structured logging!
    rotation="10 MB", # Rotates the log file when it reaches 10 MB
    catch=True # Catches exceptions to prevent crashes in logging
)

logger.info("Logger configured. Starting the Traceable Pipe v3.")


logger.info("Loading Silero VAD model from torch.hub...")
try:
    # trust_repo=True is needed to suppress the warning and run non-interactively
    # We unpack the main tuple of (model, utils_tuple)
    silero_model, utils_tuple = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    
    # Now, we unpack the nested tuple of utility functions
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils_tuple
    logger.success("Silero VAD model and utilities loaded successfully.")

except Exception as e:
    logger.critical(f"Failed to load Silero VAD model: {e}")
    # Exit if VAD can't be loaded, as it's critical
    sys.exit(1)

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.stream = None
        self.audio_queue = queue.Queue() # A thread-safe queue to pass audio chunks

    def _audio_callback(self, indata, frames, time, status):
        """This is called by the sounddevice stream for each new audio chunk."""
        self.audio_queue.put(indata.copy())

    def start_recording(self):
        """Starts the non-blocking audio stream."""
        if self.stream is not None:
            self.stop_recording() # Ensure any old stream is closed

        logger.info("Starting audio stream...")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback
        )
        self.stream.start()

    def stop_recording(self) -> np.ndarray:
        """Stops the stream and returns the full recorded audio."""
        if self.stream is None:
            return np.array([], dtype=np.float32)

        logger.info("Stopping audio stream...")
        self.stream.stop()
        self.stream.close()
        self.stream = None

        # Drain the queue and concatenate all chunks
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())
        
        if not audio_chunks:
            return np.array([], dtype=np.float32)
            
        return np.concatenate(audio_chunks, axis=0)


# --- 1.2: SPEECH-TO-TEXT SUB-SYSTEM ---
class Transcriber:
    def __init__(self, model_path="models/whisper-large-v3-turbo"):
        self.model_path = model_path
        logger.info(f"Transcriber initialized with model path: '{self.model_path}'.")
        # Note: The model is loaded on-demand by the transcribe function.
        # For a production system, we would pre-load this.

    def transcribe_audio(self, audio_filepath):
        logger.info(f"Starting transcription for '{audio_filepath}'...")
        
        # --- TRACEPOINT START ---
        start_time = time.time()
        
        try:
            logger.debug("Forcing transcription to English (language='en').")
            # mlx_whisper can take a file path directly
            result = mlx_whisper.transcribe(
                audio=audio_filepath,
                path_or_hf_repo=self.model_path,
                language="en",
                word_timestamps=False # Keep it simple for now
            )
            transcribed_text = result["text"].strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

        # --- TRACEPOINT END ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.success(f"Transcription complete in {elapsed_time:.2f}s.")
        logger.info(f"Transcribed Text: '{transcribed_text}'")
        
        return transcribed_text


# --- 1.3: LLM COGNITIVE SUB-SYSTEM ---
class LLMEngine:
    def __init__(self, llm_instance):
        if llm_instance is None:
            raise ValueError("LLM instance cannot be None.")
        self.llm = llm_instance
        logger.success("LLMEngine initialized with pre-loaded model.")

    async def generate_response(self, user_prompt: str, text_queue: asyncio.Queue):
        # --- THIS METHOD IS NOW A GENERATOR ---
        
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a helpful, brief, and conversational AI assistant. Your responses must be broken into complete sentences, ending with proper punctuation with no emojis. This is critical for the text-to-speech system.",
            },
            {"role": "user", "content": user_prompt},
        ]
        
        logger.info(f"Starting LLM stream for prompt: '{user_prompt}'")
        # --- TRACEPOINT START ---
        start_time = time.time()

        # The blocking call will be run in a separate thread.
        loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(
            None,  # Use the default thread pool executor
            lambda: self.llm.create_chat_completion(
                messages=prompt_messages, max_tokens=150, stream=True
            )
        )
        
        full_response_text = ""
        # Iterate over the stream of chunks
        for chunk in stream:
            # Extract the token from the chunk
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                token = delta['content']
                full_response_text += token
                await text_queue.put(token) # Put token into the async queue
       
        await text_queue.put(None) # Sentinel value to signal the end of the stream
        logger.success(f"LLM stream finished in {time.time() - start_time:.2f}s.")

        


class TextToSpeechEngine:
    """
    A stateless, asynchronous TTS engine that wraps the high-level 
    `generate_audio` function from the mlx-audio library.
    """
    # Path to the directory containing all Kokoro model files
    LOCAL_MODEL_PATH = "models/Kokoro"
    
    def __init__(self, speaker="af_heart"):
        self.SPEAKER = speaker
        logger.info(f"TextToSpeechEngine initialized to use model path '{self.LOCAL_MODEL_PATH}'.")
        # No pre-loading is done. We rely on the library's internal caching.

    async def synthesize_speech(self, text_chunk: str, output_prefix: str) -> str | None:
        """
        Synthesizes a chunk of text by calling the library's high-level function.
        """
        logger.info(f"Synthesizing chunk: '{text_chunk}'")
        loop = asyncio.get_running_loop()
        
        # We will run the known-good, high-level function in an executor.
        def _blocking_synthesis():
            try:
                # This is the same successful pattern from our sanity check.
                generate_audio(
                    text=text_chunk,
                    model_path=self.LOCAL_MODEL_PATH,
                    voice=self.SPEAKER,
                    file_prefix=output_prefix,
                    verbose=False, # Keep our logs clean
                    play=False
                )

                # Find the generated file.
                import glob
                files = glob.glob(f"{output_prefix}*.wav")
                if files:
                    return files[0]
                else:
                    logger.error(f"generate_audio ran but no output file was found for prefix '{output_prefix}'")
                    return None
            except Exception as e:
                logger.error(f"Exception during blocking synthesis: {e}")
                return None

        audio_file = await loop.run_in_executor(None, _blocking_synthesis)
        
        if audio_file:
            logger.success(f"Synthesis complete for chunk, saved to '{audio_file}'")
            return audio_file
        else:
            logger.error(f"Synthesis failed for chunk: '{text_chunk}'")
            return None

class AudioPlayer:
    def __init__(self, sample_rate=24000): # Kokoro's default sample rate
        self.sample_rate = sample_rate
        self._playback_queue = asyncio.Queue()
        self._playback_task = None
        self._stream = None # This will hold our persistent audio stream
        self._stop_event = threading.Event()
        logger.info("AudioPlayer initialized for persistent stream playback.")

    def start(self):
        """Initializes the audio stream and starts the playback loop."""
        # --- THE FIX: Create ONE stream that lives forever ---
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            self._stream.start()
            self._playback_task = asyncio.create_task(self._playback_loop())
            logger.success("AudioPlayer stream started and playback loop is running.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to open audio stream. Error: {e}")

    async def stop(self):
        """Stops the playback loop and closes the audio stream gracefully."""
        await self._playback_queue.put(None)
        if self._playback_task:
            await self._playback_task
        
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
        
        # sd.stop() is thread-safe and will interrupt sd.wait()
        sd.stop()
        logger.info("Audio stream stopped immediately.")

    def _blocking_play(self, audio_filepath):
        try:
            sample_rate, audio_data = read(audio_filepath)
            sd.play(audio_data, sample_rate)
            sd.wait() # This will be interrupted by sd.stop()
        except Exception as e:
            logger.error(f"Playback failed for {audio_filepath}: {e}")
    

    async def _playback_loop(self):
        """The core loop that reads files, CONVERTS DTYPE, and WRITES to the persistent stream."""
        loop = asyncio.get_running_loop()
        
        while True:
            audio_filepath = await self._playback_queue.get()
            if audio_filepath is None:
                break
            
            logger.info(f"Player loop processing: '{audio_filepath}'")
            
            try:
                # Read the audio file (likely as int16)
                sample_rate, audio_data = await loop.run_in_executor(
                    None, read, audio_filepath
                )
                if sample_rate != self.sample_rate:
                    logger.warning(f"Audio file sample rate ({sample_rate}) differs from stream rate ({self.sample_rate}).")
                
                # --- THE FIX: DTYPE CONVERSION AND NORMALIZATION ---
                # Check if the data is int16
                if audio_data.dtype == np.int16:
                    # Convert to float32 and normalize from [-32768, 32767] to [-1.0, 1.0]
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    logger.debug("Converted audio data from int16 to float32 for playback.")
                # --- END OF FIX ---

            except Exception as e:
                logger.error(f"Failed to read or convert audio file {audio_filepath}: {e}")
                continue

            if self._stream:
                try:
                    await loop.run_in_executor(
                        None, self._stream.write, audio_data
                    )
                    logger.success(f"Player loop finished writing: '{audio_filepath}'")
                except Exception as e:
                    logger.error(f"Error writing to audio stream: {e}")



# --- STATES AND EVENTS ---
class AgentState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()

# class UserEvent(Enum):
#     PUSH_TO_TALK_START = auto()
#     PUSH_TO_TALK_STOP = auto()

# --- THE  STATEFUL VOICE AGENT ---
class VoiceAgent:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        # State Machine Core
        self.state = AgentState.IDLE
        # self.event_queue = asyncio.Queue() is NO LONGER NEEDED.

        self.processing_task = None # To keep track of the running pipeline task
        
        # Load Models & Initialize Sub-systems
        self.llm_instance = self._load_llm()
        # self.recorder = AudioRecorder()
        self.transcriber = Transcriber()
        self.llm_engine = LLMEngine(self.llm_instance)
        self.tts_engine = TextToSpeechEngine()
        self.player = AudioPlayer()
         # --- NEW AUDIO PIPELINE SERVICES ---
        self.raw_audio_queue = asyncio.Queue()
        self.user_utterance_queue = asyncio.Queue()
        self.audio_streamer = AudioStreamer(self.raw_audio_queue, loop)
        self.vad_processor = VADProcessor(self.raw_audio_queue, self.user_utterance_queue)
        self.currently_speaking_utterance = "" # For echo checking

       
        logger.info(f"Voice Agent initialized. Initial state: {self.state.name}")

    def is_echo(self, user_text: str) -> bool:
        """A simple software-based echo cancellation check."""
        if not self.currently_speaking_utterance:
            return False
            
        # Normalize both strings
        agent_norm = ''.join(c for c in self.currently_speaking_utterance if c.isalnum()).lower()
        user_norm = ''.join(c for c in user_text if c.isalnum()).lower()

        if not user_norm: # Ignore empty transcriptions
            return True 

        # Check if the user's speech is just a small fragment of the agent's speech
        if user_norm in agent_norm:
            logger.warning(f"Echo detected. User said '{user_text}', which is a substring of agent's speech.")
            return True
            
        return False

    async def handle_barge_in(self, interrupting_audio_data: np.ndarray):
        """The core interruption logic."""
        logger.warning("BARGE-IN TRIGGERED!")
        
        # 1. Stop the current response
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            logger.info("Cancelled in-flight processing task.")
        await self.player.interrupt()
        
        # 2. Immediately start processing the new input
        await self.transition_to(AgentState.PROCESSING)
        self.processing_task = asyncio.create_task(self.run_pipeline(interrupting_audio_data))


    async def transition_to(self, new_state: AgentState):
        logger.info(f"State transition: {self.state.name} -> {new_state.name}")
        self.state = new_state
    
    # --- The Core Pipeline Logic ---
    async def run_pipeline(self, audio_data: np.ndarray):
        """
        The full STT -> LLM -> TTS pipeline, run for a single user utterance.
        Now accepts audio data directly, no file I/O needed for STT.
        """
        logger.info("Starting audio processing pipeline...")
        loop = asyncio.get_running_loop()
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        try:
            # --- FIX 1: Transcribe directly from the NumPy array ---
            # We no longer need to write to a file first.
            # We also get the sample rate from our VADProcessor.
            logger.info("Transcribing audio data directly from memory...")
            transcribed_text = await loop.run_in_executor(
                None, 
                lambda: mlx_whisper.transcribe(
                    audio=audio_data, # Pass the NumPy array
                    path_or_hf_repo=self.transcriber.model_path,
                    language="en"
                )["text"].strip()
            )
            
            if not transcribed_text:
                raise ValueError("Transcription failed or produced no text.")
            
            logger.success(f"Transcription complete: '{transcribed_text}'")

            # Set state to SPEAKING.
            await self.transition_to(AgentState.SPEAKING)
            llm_response_text = ""

            # Setup async queues for the streaming pipeline
            text_token_queue = asyncio.Queue()
            tts_sentence_queue = asyncio.Queue()
            full_text_future = asyncio.Future()

            
            # Create and run all concurrent data-processing tasks
            llm_task = asyncio.create_task(
                self.llm_engine.generate_response(transcribed_text, text_token_queue)
            )
            chunker_task = asyncio.create_task(
                self.text_chunker(text_token_queue, tts_sentence_queue)
            )
            tts_task = asyncio.create_task(
                self.tts_consumer(tts_sentence_queue, timestamp)
            )
            
            await asyncio.gather(llm_task, chunker_task, tts_task)
            
            logger.info("Data processing complete. Waiting for playback to finish...")
            while not self.player._playback_queue.empty():
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
        
        finally:
            await self.transition_to(AgentState.IDLE)
            logger.info("Pipeline complete. Returning to IDLE.")

    # The tts_consumer now just adds to the player's queue
    async def tts_consumer(self, tts_queue: asyncio.Queue, timestamp: str):
        chunk_index = 0
        while True:
            text_chunk = await tts_queue.get()
            if text_chunk is None:
                 break
            
            output_prefix = f"tts_output_{timestamp}_{chunk_index:02d}"
            audio_file = await self.tts_engine.synthesize_speech(text_chunk, output_prefix)
            await self.player.add_to_queue(audio_file) # Add to player's queue
            chunk_index += 1

    async def text_chunker(self, text_queue: asyncio.Queue, tts_queue: asyncio.Queue):
        """A task that consumes tokens, forms sentences, and passes them to the TTS queue."""
        sentence_buffer = ""
        sentence_terminators = {".", "?", "!"}
        
        while True:
            token = await text_queue.get()
            if token is None: # End of stream
                if sentence_buffer.strip():
                    await tts_queue.put(sentence_buffer.strip())
                await tts_queue.put(None)
                break
            
            print(token, end="", flush=True) # Visualize the stream
            sentence_buffer += token
            if any(term in token for term in sentence_terminators):
                await tts_queue.put(sentence_buffer.strip())
                sentence_buffer = ""
    
    def _load_llm(self):
        # This private method centralizes the heavy LLM loading.
        logger.info("Loading LLM... (This may take a moment)")
        try:
            from llama_cpp import Llama
            llm = Llama(
                model_path="./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                n_gpu_layers=-1,
                n_ctx=32768,
                verbose=False
            )
            logger.success("LLM loaded into memory.")
            return llm
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load LLM. Agent cannot start. Error: {e}")
            return None

    

    async def start_services(self):
        """Starts all long-running background tasks."""
        self.player.start()
        self.audio_streamer.start()
        await self.vad_processor.start() # vad_processor.start is async

    async def stop_services(self):
        """Stops all long-running background tasks."""
        await self.player.stop()
        self.audio_streamer.stop()
        await self.vad_processor.stop()


    async def start_conversation_turn(self):
        # This is now the main async orchestrator for a single turn.
        if not self.llm_instance:
             return
            
        logger.info("--- STARTING NEW ASYNC CONVERSATION TURN ---")
        
        # 1. & 2. Record and Transcribe (
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        recording_file = f"temp_recording_{timestamp}.wav"
        self.recorder.record_audio(output_filename=recording_file)
        transcribed_text = self.transcriber.transcribe_audio(recording_file)
        if not transcribed_text:
             return
        
        # 3. Create the communication queues
        text_token_queue = asyncio.Queue()
        tts_sentence_queue = asyncio.Queue()
        # audio_playback_queue = asyncio.Queue()
        
        # 4. Create and run all tasks concurrently
        llm_task = asyncio.create_task(
            self.llm_engine.generate_response(transcribed_text, text_token_queue)
        )
        chunker_task = asyncio.create_task(
            self.text_chunker(text_token_queue, tts_sentence_queue)
        )
        tts_task = asyncio.create_task(
            self.tts_consumer(tts_sentence_queue, timestamp) # Simplified
        )
        
        # 5. Wait for the entire pipeline to complete
        await asyncio.gather(llm_task, chunker_task, tts_task)
        
        logger.info("--- ASYNC CONVERSATION TURN COMPLETE ---")
        # Note: Playback may still be ongoing in the background.
    
   # The main run loop is now the final orchestrator
    async def run(self):
        await self.start_services()
        logger.info("Agent run loop started. Listening for speech...")

        while True:
            user_audio_data = await self.user_utterance_queue.get()

            if self.state == AgentState.IDLE or self.state == AgentState.PROCESSING:
                if self.state == AgentState.PROCESSING and self.processing_task and not self.processing_task.done():
                    logger.info("User spoke again while processing, queuing up next turn.")
                    # More advanced logic could go here, for now we let the current turn finish.
                    # This prevents rapid-fire interruptions.
                    continue

                logger.info("User utterance detected while agent is idle/processing. Starting new turn.")
                await self.transition_to(AgentState.PROCESSING)
                self.processing_task = asyncio.create_task(self.run_pipeline(user_audio_data))
            
            elif self.state == AgentState.SPEAKING:
                logger.info("User utterance detected while agent is speaking...")
                # We need to transcribe this small chunk to see if it's echo or barge-in
                loop = asyncio.get_running_loop()
                user_text = await loop.run_in_executor(
                    None,
                    lambda: mlx_whisper.transcribe(
                        audio=user_audio_data,
                        path_or_hf_repo=self.transcriber.model_path,
                        language="en"
                    )["text"].strip()
                )

                if user_text and not self.is_echo(user_text):
                    # It's a real interruption!
                    await self.handle_barge_in(user_audio_data)
                else:
                    # It's echo or noise, ignore it.
                    pass


# --- KEYBOARD LISTENER SETUP ---
# class KeyboardListener:
#     def __init__(self, event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
#         self.event_queue = event_queue
#         self.loop = loop
#         self.listener_thread = None
#         self.is_space_pressed = False

#     def on_press(self, key):
#         if key == keyboard.Key.space and not self.is_space_pressed:
#             self.is_space_pressed = True
#             # Use call_soon_threadsafe to safely interact with the asyncio loop
#             self.loop.call_soon_threadsafe(
#                 self.event_queue.put_nowait, UserEvent.PUSH_TO_TALK_START
#             )

#     def on_release(self, key):
#         if key == keyboard.Key.space:
#             self.is_space_pressed = False
#             self.loop.call_soon_threadsafe(
#                 self.event_queue.put_nowait, UserEvent.PUSH_TO_TALK_STOP
#             )
            
#     def start(self):
#         # The pynput listener runs in its own thread
#         listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
#         self.listener_thread = threading.Thread(target=listener.run, daemon=True)
#         self.listener_thread.start()
#         logger.info("Keyboard listener started in a background thread.")


class AudioStreamer:
    """Uses PyAudio to continuously stream audio in a background thread."""
    def __init__(self, raw_audio_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, sample_rate=16000, frame_duration_ms=30):
        self.raw_audio_queue = raw_audio_queue
        # --- THE FIX: Store a reference to the main event loop ---
        self.loop = loop
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.chunk_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_running = threading.Event()

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """Callback to put raw audio data into our async queue."""
        # --- THE FIX: Use the stored loop reference ---
        self.loop.call_soon_threadsafe(self.raw_audio_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    def start(self):
        """Starts the audio streaming in a dedicated background thread."""
        logger.info("Starting PyAudio streamer...")
        self.is_running.set()
        thread = threading.Thread(target=self._run_stream, daemon=True)
        thread.start()
        logger.success("PyAudio streamer thread started.")

    def _run_stream(self):
        """This function runs in its own dedicated thread."""
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._stream_callback
        )
        while self.stream.is_active() and self.is_running.is_set():
            time.sleep(0.1)

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        logger.info("PyAudio stream resources released.")

    def stop(self):
        """Signals the streaming thread to stop."""
        logger.info("Stopping PyAudio streamer...")
        self.is_running.clear()

class VADProcessor:
    """
    A two-stage VAD that uses a lightweight WebRTCVAD for initial filtering
    and a more powerful Silero VAD for final confirmation.
    """
    def __init__(self, raw_audio_queue: asyncio.Queue, utterance_queue: asyncio.Queue, sample_rate=16000, frame_duration_ms=30, padding_ms=300):
        self.raw_audio_queue = raw_audio_queue
        self.utterance_queue = utterance_queue
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        
        # WebRTCVAD setup
        self.vad = webrtcvad.Vad(3) # Aggressiveness level 3
        
        # Ring buffer logic from the example
        num_padding_frames = padding_ms // frame_duration_ms
        self.ring_buffer = collections.deque(maxlen=num_padding_frames)
        self.ratio = 0.75 # Ratio of speech frames to trigger
        
        # Silero VAD setup
        self.silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        
        self.vad_task = None

    def is_speech(self, frame: bytes) -> bool:
        """Uses WebRTCVAD to check a single frame for speech."""
        return self.vad.is_speech(frame, self.sample_rate)

    async def start(self):
        logger.info("Starting Two-Stage VAD processor...")
        self.vad_task = asyncio.create_task(self._vad_collector())

    async def stop(self):
        logger.info("Stopping VAD processor...")
        if self.vad_task:
            self.vad_task.cancel()
            try:
                 await self.vad_task
            except asyncio.CancelledError:
                 pass
        logger.success("VAD processor stopped.")

    async def _vad_collector(self):
        """
        Consumes raw audio and collects utterances based on WebRTCVAD,
        then confirms them with Silero VAD.
        """
        triggered = False
        voiced_frames = []
        
        while True:
            try:
                frame = await self.raw_audio_queue.get()

                if len(frame) != int(self.sample_rate * self.frame_duration_ms / 1000 * 2): # 2 bytes per sample
                    continue
                
                is_speech_frame = self.is_speech(frame)

                if not triggered:
                    self.ring_buffer.append((frame, is_speech_frame))
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])
                    if num_voiced > self.ratio * self.ring_buffer.maxlen:
                        logger.debug("WebRTCVAD triggered.")
                        triggered = True
                        for f, s in self.ring_buffer:
                            voiced_frames.append(f)
                        self.ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    self.ring_buffer.append((frame, is_speech_frame))
                    num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                    if num_unvoiced > self.ratio * self.ring_buffer.maxlen:
                        logger.debug("WebRTCVAD un-triggered.")
                        triggered = False
                        
                        # --- Utterance complete, now confirm with Silero ---
                        utterance_bytes = b''.join(voiced_frames)
                        self.ring_buffer.clear()
                        voiced_frames = []
                        
                        await self.confirm_with_silero(utterance_bytes)
            except asyncio.CancelledError:
                break

    async def confirm_with_silero(self, audio_bytes: bytes):
        """Runs the collected utterance through Silero VAD for final confirmation."""
        logger.info("WebRTCVAD detected utterance, confirming with Silero VAD...")
        
        # Convert bytes to float32 tensor for Silero
        audio_int16 = np.frombuffer(audio_bytes, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float32)

        # Run Silero VAD
        loop = asyncio.get_running_loop()
        speech_timestamps = await loop.run_in_executor(
            None, lambda: get_speech_timestamps(audio_tensor, self.silero_model, sampling_rate=self.sample_rate) 
        )

        if len(speech_timestamps) > 0:
            logger.success("Silero VAD confirmed speech. Emitting utterance.")
            # We already have the float32 data, so we can put it directly
            await self.utterance_queue.put(audio_float32)
        else:
            logger.warning("Silero VAD detected NOISE. Discarding utterance.")

# --- MAIN EXECUTION BLOCK ---
async def main():
    loop = asyncio.get_running_loop()
    agent = VoiceAgent(loop)
    try:
        # The main function now just starts the agent's run loop.
        await agent.run()
    finally:
        logger.info("Shutting down agent services...")
        await agent.stop_services()
        logger.success("Agent shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down.")
    
   