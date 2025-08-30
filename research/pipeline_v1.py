
# pipeline_v1.py

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
    # We use Llama 3 8B Q4_K_M as validated in Phase 0
    LLM_MODEL_PATH = "./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    
    def __init__(self, n_ctx=32768): 
        # Using a context window (n_ctx) of 4096, a good balance 
        # based on our Phase 0 architectural note.
        self.n_ctx = n_ctx
        
        # We perform the heavy model loading in __init__ to avoid loading 
        # it every time we run inference.
        logger.info(f"LLMEngine initializing and loading model: {self.LLM_MODEL_PATH}")
        start_load_time = time.time()
        
        try:
            self.llm = Llama(
                model_path=self.LLM_MODEL_PATH,
                n_gpu_layers=-1, # Offload all possible layers to the M4 GPU
                n_ctx=self.n_ctx, 
                verbose=False # Keep the Llama library quiet
            )
            load_time = time.time() - start_load_time
            logger.success(f"LLM loaded successfully in {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self.llm = None

    def generate_response(self, user_prompt: str) -> str:
        if not self.llm:
            return "LLM service unavailable."

        # Define the conversational context using the Llama 3 chat template
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a helpful, brief, and conversational AI assistant. Keep your answers short and to the point, suitable for a real-time voice interaction.",
            },
            {"role": "user", "content": user_prompt},
        ]
        
        logger.info(f"Starting LLM inference for prompt: '{user_prompt}'")
        # --- TRACEPOINT START ---
        start_time = time.time()
        
        try:
            # We are using the synchronous create_chat_completion for this dumb pipe
            output = self.llm.create_chat_completion(
                messages=prompt_messages, # type: ignore
                max_tokens=150, # Keep responses short for speed
                stream=False    # Crucial for a synchronous pipeline test
            )
            
            response_text = output['choices'][0]['message']['content'].strip() # type: ignore
            
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            return "I am currently unable to process your request."

        # --- TRACEPOINT END ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate tokens per second (a key metric)
        # We can't easily count output tokens without streaming, so we'll approximate.
        # The internal Llama output logs would be more precise, but we keep verbose=False for cleanliness.
        logger.success(f"LLM inference complete in {elapsed_time:.2f}s.")
        logger.info(f"LLM Response Text: '{response_text}'")
        
        return response_text


# --- 1.4: TEXT-TO-SPEECH SUB-SYSTEM ---
class TextToSpeechEngine:
    TTS_MODEL_PATH = "models/Kokoro"
    SPEAKER = "af_heart"
    
    def __init__(self):
        logger.info(f"TextToSpeechEngine initialized with model path: '{self.TTS_MODEL_PATH}' and speaker: '{self.SPEAKER}'.")
        # Note: The model is loaded on-demand by the generate_audio function.
        # This will contribute to the synthesis time in this version of the pipeline.
        
    def synthesize_speech(self, text_to_speak: str, output_filename_prefix="tts_output"):
        logger.info(f"Starting speech synthesis for text: '{text_to_speak}'")
        
        # --- TRACEPOINT START ---
        start_time = time.time()
        
        try:
            generate_audio(
                text=text_to_speak,
                model_path=self.TTS_MODEL_PATH,
                voice=self.SPEAKER,
                file_prefix=output_filename_prefix,
                audio_format="wav",
                join_audio=False,
                stream=False,
                play=False,
                verbose=False # Keep our own logs clean
            )
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return None
        
        # --- TRACEPOINT END ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # The function creates a file like "prefix_000.wav", we need to find it
        import glob
        potential_files = glob.glob(f"{output_filename_prefix}*.wav")
        if not potential_files:
            logger.error("TTS output file not found after synthesis.")
            return None
            
        output_filepath = potential_files[0]
        
        logger.success(f"Speech synthesis complete in {elapsed_time:.2f}s. Audio saved to '{output_filepath}'.")
        
        return output_filepath

# --- 1.5: AUDIO PLAYBACK SUB-SYSTEM ---
class AudioPlayer:
    def play_audio(self, audio_filepath: str):
        if not audio_filepath:
            logger.warning("No audio file path provided to player.")
            return
            
        logger.info(f"Starting playback of '{audio_filepath}'...")
        # --- TRACEPOINT START ---
        start_time = time.time()
        
        try:
            sample_rate, audio_data = read(audio_filepath)
            sd.play(audio_data, sample_rate)
            sd.wait() # Block until playback is finished
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return

        # --- TRACEPOINT END ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.success(f"Playback finished. Duration: {elapsed_time:.2f}s.")

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

    def start_conversation_turn(self):
        """
        Executes a single, synchronous turn of the conversation.
        This contains the logic from our v1 main block.
        """
        if not self.llm_instance:
            logger.error("LLM not loaded. Cannot start conversation.")
            return

        logger.info("--- STARTING NEW CONVERSATION TURN ---")
        turn_start_time = time.time()
        
        # The logic is identical to v1, but uses the pre-initialized class members.
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 1. Record Audio
        recording_file = f"temp_recording_{timestamp}.wav"
        self.recorder.record_audio(output_filename=recording_file)
        
        # 2. Transcribe
        transcribed_text = self.transcriber.transcribe_audio(recording_file)
        if not transcribed_text: return
            
        # 3. Generate Response
        llm_response = self.llm_engine.generate_response(transcribed_text)
        if not llm_response: return
            
        # 4. Synthesize Speech
        tts_file = self.tts_engine.synthesize_speech(llm_response, f"tts_output_{timestamp}")
        if not tts_file: return
            
        # 5. Play Audio
        self.player.play_audio(tts_file)
        
        turn_end_time = time.time()
        total_turn_latency = turn_end_time - turn_start_time
        logger.info(f"--- CONVERSATION TURN COMPLETE ---")
        logger.info(f"Total Turn Latency: {total_turn_latency:.2f}s")
        


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    logger.info("--- STARTING PIPELINE V1 EXECUTION ---")
    pipeline_start_time = time.time()
    
    # Instantiate all engines at startup
    llm_engine = LLMEngine()
    tts_engine = TextToSpeechEngine()
    audio_player = AudioPlayer()
    
    # --- LEG 1 & 2: STT ---
    recorder = AudioRecorder(duration=4) # A bit more time for a question
    transcriber = Transcriber()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    recording_file = f"temp_recording_{timestamp}.wav"
    recorded_audio, sample_rate = recorder.record_audio(output_filename=recording_file)
    
    if recorded_audio is None:
        logger.critical("Audio input failed. Exiting pipeline.")
        sys.exit(1)

    transcribed_text = transcriber.transcribe_audio(recording_file)
    
    if transcribed_text is None:
        logger.critical("STT failed. Exiting pipeline.")
        sys.exit(1)
        
    logger.success("STT Leg Complete.")

    # --- LEG 3: LLM ---
    llm_response = llm_engine.generate_response(transcribed_text)

    if not llm_response:
        logger.critical("LLM leg failed. Exiting pipeline.")
        sys.exit(1)
        
    logger.success("LLM Leg Complete.")
        
    # --- LEG 4 & 5: TTS & Playback ---
    tts_output_file = tts_engine.synthesize_speech(llm_response, output_filename_prefix=f"tts_output_{timestamp}")
    
    if tts_output_file:
        logger.success("TTS Leg Complete.")
        audio_player.play_audio(tts_output_file)
        logger.success("Playback Complete.")
    else:
        logger.critical("TTS leg failed. Exiting pipeline.")
        sys.exit(1)

    pipeline_end_time = time.time()
    total_pipeline_latency = pipeline_end_time - pipeline_start_time
    logger.info("--- PIPELINE V1 EXECUTION COMPLETE ---")
    logger.info(f"Total End-to-End Latency: {total_pipeline_latency:.2f} seconds.")