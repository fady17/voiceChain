import asyncio
import sys
import time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from llama_cpp import Llama
import mlx_whisper

from loguru import logger
# New imports for the final leg
from mlx_audio.tts.generate import generate_audio


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
    def __init__(self, sample_rate=16000, duration=5):
        self.sample_rate = sample_rate
        self.duration = duration
        logger.info(f"AudioRecorder initialized with {self.sample_rate} Hz sample rate and {self.duration}s duration.")

    def record_audio(self, output_filename="temp_recording.wav"):
        """
        Records audio from the default microphone for a fixed duration.
        """
        logger.info(f"Starting {self.duration}-second audio recording...")
        
        # --- TRACEPOINT START ---
        start_time = time.time()
        
        # The actual recording command. It's a blocking call.
        recording_data = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait() # Wait until recording is finished

        # --- TRACEPOINT END ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.success(f"Recording complete. Actual time elapsed: {elapsed_time:.2f}s.")
        logger.debug(f"Recording data shape: {recording_data.shape}, dtype: {recording_data.dtype}")

        # Save the recording to a file for debugging and for the STT engine
        try:
            write(output_filename, self.sample_rate, recording_data)
            logger.info(f"Recording saved to '{output_filename}'.")
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return None, None

        return recording_data, self.sample_rate


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

        


# --- 1.4: TEXT-TO-SPEECH SUB-SYSTEM ---
class TextToSpeechEngine:
    TTS_MODEL_PATH = "models/Kokoro"
    SPEAKER = "af_heart"
    
    def __init__(self):
        logger.info(f"TextToSpeechEngine initialized with model path: '{self.TTS_MODEL_PATH}' and speaker: '{self.SPEAKER}'.")
        # Note: The model is loaded on-demand by the generate_audio function.
        # This will contribute to the synthesis time in this version of the pipeline.
        
    async def synthesize_speech(self, text_chunk: str, output_prefix: str) -> str:
        logger.info(f"Synthesizing chunk: '{text_chunk}'")
        loop = asyncio.get_running_loop()
        
        # Run the blocking TTS function in an executor
        await loop.run_in_executor(
            None, 
            lambda: generate_audio(
                text=text_chunk, model_path=self.TTS_MODEL_PATH, voice=self.SPEAKER,
                file_prefix=output_prefix, verbose=False, play=False
            )
        )
        
        import glob
        files = glob.glob(f"{output_prefix}*.wav")
        if files:
            return files[0]
        return None # type: ignore

class AudioPlayer:
    def __init__(self):
        self._playback_queue = asyncio.Queue()
        self._playback_task = None
        self._is_playing = asyncio.Event() # Event to signal playback status
        logger.info("AudioPlayer initialized with internal queue.")

    def start(self):
        """Starts the dedicated playback loop as a background task."""
        self._playback_task = asyncio.create_task(self._playback_loop())
        logger.info("AudioPlayer playback loop started.")

    async def stop(self):
        """Stops the playback loop gracefully."""
        await self._playback_queue.put(None) # Sentinel to stop the loop
        if self._playback_task:
            await self._playback_task
        logger.info("AudioPlayer playback loop stopped.")
    
    async def add_to_queue(self, audio_filepath: str):
        """A thread-safe way to add audio to be played."""
        if audio_filepath:
            await self._playback_queue.put(audio_filepath)

    async def _playback_loop(self):
        """The core loop that plays audio sequentially from the queue."""
        loop = asyncio.get_running_loop()
        
        while True:
            audio_filepath = await self._playback_queue.get()
            if audio_filepath is None:
                break # Exit condition
            
            logger.info(f"Player loop processing: '{audio_filepath}'")
            self._is_playing.set() # Signal that we are now busy

            # Run the blocking playback in an executor
            await loop.run_in_executor(
                None, self._blocking_play, audio_filepath
            )
            
            self._is_playing.clear() # Signal that we are now free
            logger.success(f"Player loop finished: '{audio_filepath}'")

    def _blocking_play(self, audio_filepath):
        try:
            from scipy.io.wavfile import read
            import sounddevice as sd
            sample_rate, audio_data = read(audio_filepath)
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Playback failed for {audio_filepath}: {e}")
            


# --- THE MAIN AGENT CLASS ---
class VoiceAgent:
    def __init__(self):
        logger.info("Initializing Voice Agent...")
        
        # --- 1. LOAD ALL MODELS AT STARTUP ---
        # This is the core of our persistent architecture.
        self.llm_instance = self._load_llm()
        
        # --- 2. INITIALIZE ALL SUB-SYSTEMS ---
        # We pass the pre-loaded models to our engine classes.
        self.recorder = AudioRecorder(duration=4)
        self.transcriber = Transcriber()
        self.llm_engine = LLMEngine(self.llm_instance) # type: ignore
        self.tts_engine = TextToSpeechEngine()
        self.player = AudioPlayer()
        
        logger.success("Voice Agent initialized successfully. All models loaded.")

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

    # The tts_consumer now just adds to the player's queue
    async def tts_consumer(self, tts_queue: asyncio.Queue, timestamp: str):
        chunk_index = 0
        while True:
            text_chunk = await tts_queue.get()
            if text_chunk is None: break
            
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
        if not self.llm_instance: return
            
        logger.info("--- STARTING NEW ASYNC CONVERSATION TURN ---")
        
        # 1. & 2. Record and Transcribe (
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        recording_file = f"temp_recording_{timestamp}.wav"
        self.recorder.record_audio(output_filename=recording_file)
        transcribed_text = self.transcriber.transcribe_audio(recording_file)
        if not transcribed_text: return
        
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



# --- MAIN EXECUTION BLOCK (The Orchestrator) ---
async def main():
    agent = VoiceAgent()
    
    # Start background services ONCE
    await agent.start_services()
    
    try:
        for i in range(2):
            logger.info(f"\n--- Triggering Conversation Turn {i+1} ---")
            input("Press Enter to start recording...")
            await agent.start_conversation_turn()
            
            # --- FIX 5: Explicitly wait for playback to finish before looping ---
            # This ensures one conversation is fully "spoken" before the next begins.
            logger.info("Waiting for audio playback to complete...")
            while not agent.player._playback_queue.empty() or agent.player._is_playing.is_set():
                await asyncio.sleep(0.1)
            logger.success("Playback complete. Ready for next turn.")

    finally:
        # Stop background services ONCE at the very end
        logger.info("Shutting down agent services...")
        await agent.stop_services()
        logger.success("Agent shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down.")