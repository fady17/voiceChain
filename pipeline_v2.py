
# pipeline_v2.py

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

    def generate_response(self, user_prompt: str):
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
        
        full_response_text = ""
        
        # try:
            # Set stream=True to get a generator
        stream = self.llm.create_chat_completion(
            messages=prompt_messages,
            max_tokens=150,
            stream=True
        )
        
        # Iterate over the stream of chunks
        for chunk in stream:
            # Extract the token from the chunk
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                token = delta['content']
                full_response_text += token
                yield token # YIELD the token to the caller
                
        # except Exception as e:
        #     logger.error(f"LLM stream failed: {e}")
        #     yield "I am currently unable to process your request."
        #     return

        # --- TRACEPOINT END ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.success(f"LLM stream finished in {elapsed_time:.2f}s.")
        logger.info(f"Full LLM Response: '{full_response_text}'")


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

    def stream_llm_to_tts(self, llm_response_generator, timestamp):
        """
        Consumes the LLM token stream, buffers sentences, and sends them to TTS.
        This is the core of our streaming TTS logic.
        """
        logger.info("Starting to stream LLM response to TTS engine...")
        sentence_buffer = ""
        sentence_terminators = {".", "?", "!"}
        
        chunk_index = 0
        for token in llm_response_generator:
            print(token, end="", flush=True) # Continue to print for visualization
            sentence_buffer += token
            
            if any(terminator in token for terminator in sentence_terminators):
                logger.info(f"Sentence boundary detected. Synthesizing chunk {chunk_index+1}: '{sentence_buffer.strip()}'")
                
                # Synthesize this complete sentence
                tts_file = self.tts_engine.synthesize_speech(
                    sentence_buffer.strip(), 
                    f"tts_output_{timestamp}_{chunk_index:02d}"
                )
                
                # Play this chunk of audio
                self.player.play_audio(tts_file) # type: ignore
                
                # Reset for the next sentence
                sentence_buffer = ""
                chunk_index += 1
        
        # After the loop, check if there's any remaining text in the buffer
        if sentence_buffer.strip():
            logger.info(f"Synthesizing final chunk {chunk_index+1}: '{sentence_buffer.strip()}'")
            tts_file = self.tts_engine.synthesize_speech(
                sentence_buffer.strip(), 
                f"tts_output_{timestamp}_{chunk_index:02d}"
            )
            self.player.play_audio(tts_file) # type: ignore

        print() # Final newline for the console output
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
            
        # --- 3, 4, 5. Stream LLM, Synthesize, and Play ---
        # This is now one integrated, streaming operation.
        llm_response_generator = self.llm_engine.generate_response(transcribed_text)
        self.stream_llm_to_tts(llm_response_generator, timestamp)
        
        # For now, in our synchronous pipeline, we will just print the tokens as they arrive.
        # This simulates the "real-time" feel. We are not yet piping them to TTS.
        full_llm_response = ""
        for token in llm_response_generator:
            print(token, end="", flush=True) # Print token immediately
            full_llm_response += token
        
        print() # Newline after the full response is printed
        logger.success("Finished consuming LLM stream.")

        if not full_llm_response: return
        tts_file = self.tts_engine.synthesize_speech(full_llm_response, f"tts_output_{timestamp}")
        if not tts_file: return
            
        # 5. Play Audio
        self.player.play_audio(tts_file)
        
        turn_end_time = time.time()
        total_turn_latency = turn_end_time - turn_start_time
        logger.info(f"--- CONVERSATION TURN COMPLETE ---")
        logger.info(f"Total Turn Latency: {total_turn_latency:.2f}s")
        


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    agent = VoiceAgent()
    
    # Simple loop to simulate multiple conversations.
    # In a real app, this could be a "while True:" loop with a proper exit condition.
    for i in range(2):
        logger.info(f"\n--- Triggering Conversation Turn {i+1} ---")
        input("Press Enter to start recording...")
        agent.start_conversation_turn()

    logger.info("Pipeline script finished.")