#pipeline_v3_stateful.py
import asyncio
import threading
import sys
import time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write , read
# from llama_cpp import Llama
import mlx_whisper
from enum import Enum, auto
from loguru import logger
from mlx_audio.tts.generate import generate_audio
from pynput import keyboard
import queue 
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

logger.info("Logger configured. Starting the Traceable Pipe v1.")

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
            transcribed_text = result["text"].strip() # type: ignore
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

class UserEvent(Enum):
    PUSH_TO_TALK_START = auto()
    PUSH_TO_TALK_STOP = auto()

# --- THE FINAL STATEFUL VOICE AGENT ---
class VoiceAgent:
    def __init__(self):
        # State Machine Core
        self.state = AgentState.IDLE
        self.event_queue = asyncio.Queue()
        self.processing_task = None # To keep track of the running pipeline task
        
        # Load Models & Initialize Sub-systems
        self.llm_instance = self._load_llm()
        self.recorder = AudioRecorder()
        self.transcriber = Transcriber()
        self.llm_engine = LLMEngine(self.llm_instance)
        self.tts_engine = TextToSpeechEngine()
        self.player = AudioPlayer()
        logger.info(f"Voice Agent initialized. Initial state: {self.state.name}")

    async def transition_to(self, new_state: AgentState):
        logger.info(f"State transition: {self.state.name} -> {new_state.name}")
        self.state = new_state
    
    # --- The Core Pipeline Logic ---
    async def run_pipeline(self, audio_data: np.ndarray):
        """This is the full, self-contained pipeline task."""
        loop = asyncio.get_running_loop()
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        try:
            # 1. Save and Transcribe
            recording_file = f"temp_recording_{timestamp}.wav"
            await loop.run_in_executor(
                None, lambda: write(recording_file, self.recorder.sample_rate, audio_data)
            )
            transcribed_text = await loop.run_in_executor(
                None, self.transcriber.transcribe_audio, recording_file
            )
            if not transcribed_text:
                raise ValueError("Transcription failed or produced no text.")
            
            # Set state to SPEAKING. From this point, the agent is considered "responding".
            await self.transition_to(AgentState.SPEAKING)

            # 2. Setup async queues for the streaming pipeline
            text_token_queue = asyncio.Queue()
            tts_sentence_queue = asyncio.Queue()
            
            # 3. Create and run all concurrent data-processing tasks
            llm_task = asyncio.create_task(
                self.llm_engine.generate_response(transcribed_text, text_token_queue)
            )
            chunker_task = asyncio.create_task(
                self.text_chunker(text_token_queue, tts_sentence_queue)
            )
            tts_task = asyncio.create_task(
                self.tts_consumer(tts_sentence_queue, timestamp)
            )
            
            # Wait for the data processing to finish. Playback happens in the background.
            await asyncio.gather(llm_task, chunker_task, tts_task)
            
            # 4. Wait for the audio player to finish its queue
            logger.info("Data processing complete. Waiting for playback to finish...")
            while not self.player._playback_queue.empty():
                await asyncio.sleep(0.1)
            # Add a small buffer to ensure the last chunk has started playing
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
        
        finally:
            # Once everything is done, spoken or failed, return to IDLE
            await self.transition_to(AgentState.IDLE)
            logger.info("Pipeline complete. Returning to IDLE. Hold SPACE to talk.")

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
    # --- The Main Agent Event Loop ---
    async def run(self):
        """The main event loop that drives the state machine."""
        self.player.start() # Start the player as a permanent background service
        logger.info("Agent run loop started. Hold SPACE to talk.")

        while True:
            event = await self.event_queue.get()
            logger.debug(f"Event: {event.name} in State: {self.state.name}")

            if self.state == AgentState.IDLE:
                if event == UserEvent.PUSH_TO_TALK_START:
                    await self.transition_to(AgentState.LISTENING)
                    self.recorder.start_recording()

            elif self.state == AgentState.LISTENING:
                if event == UserEvent.PUSH_TO_TALK_STOP:
                    audio_data = self.recorder.stop_recording()
                    if audio_data.size < 16000: # Less than 1 second of audio
                        logger.warning("Recording too short. Returning to IDLE.")
                        await self.transition_to(AgentState.IDLE)
                        continue
                    
                    await self.transition_to(AgentState.PROCESSING)
                    # Start the entire pipeline as a single, non-blocking background task
                    self.processing_task = asyncio.create_task(self.run_pipeline(audio_data))

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
            
     # --- FIX 2: A dedicated method to start background services ---
    async def start_services(self):
        """Starts all long-running background tasks."""
        self.player.start()

    # --- FIX 3: A dedicated method to stop services gracefully ---
    async def stop_services(self):
        """Stops all long-running background tasks."""
        await self.player.stop()

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


# --- KEYBOARD LISTENER SETUP ---
class KeyboardListener:
    def __init__(self, event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.event_queue = event_queue
        self.loop = loop
        self.listener_thread = None
        self.is_space_pressed = False

    def on_press(self, key):
        if key == keyboard.Key.space and not self.is_space_pressed:
            self.is_space_pressed = True
            # Use call_soon_threadsafe to safely interact with the asyncio loop
            self.loop.call_soon_threadsafe(
                self.event_queue.put_nowait, UserEvent.PUSH_TO_TALK_START
            )

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.is_space_pressed = False
            self.loop.call_soon_threadsafe(
                self.event_queue.put_nowait, UserEvent.PUSH_TO_TALK_STOP
            )
            
    def start(self):
        # The pynput listener runs in its own thread
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener_thread = threading.Thread(target=listener.run, daemon=True)
        self.listener_thread.start()
        logger.info("Keyboard listener started in a background thread.")



# --- MAIN EXECUTION BLOCK ---
async def main():
    agent = VoiceAgent()
    
    # Get the current asyncio event loop
    main_loop = asyncio.get_running_loop()
    
    # Create and start the keyboard listener, passing it the agent's queue and the loop
    k_listener = KeyboardListener(agent.event_queue, main_loop)
    k_listener.start()
    
    # Run the agent's main loop
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")