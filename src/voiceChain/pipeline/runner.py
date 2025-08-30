import asyncio
import numpy as np
# from loguru import logger

from ..core.stt import Transcriber
from ..core.llm import LLMEngine
from ..core.tts import TextToSpeechEngine
from .services import ServiceManager

class PipelineRunner:
    """
    Orchestrates the execution of a single conversational turn:
    STT -> LLM -> TTS. It manages the flow of data between these components.
    """
    def __init__(self, transcriber: Transcriber, llm_engine: LLMEngine, tts_engine: TextToSpeechEngine, services: ServiceManager):
        self.transcriber = transcriber
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine
        self.services = services

    async def run(self, audio_data: np.ndarray) -> str:
        """
        Executes the full pipeline for a given user utterance.

        Args:
            audio_data: The user's speech as a NumPy array.

        Returns:
            The full text of the agent's response.
        """
        loop = asyncio.get_running_loop()
        
        # 1. Synchronous STT in its dedicated executor
        transcribed_text = await loop.run_in_executor(
            self.services.stt_executor, self.transcriber.transcribe_audio_sync, audio_data
        )
        if not transcribed_text:
            raise ValueError("Transcription failed or returned empty.")
        
        text_token_queue = asyncio.Queue()
        tts_sentence_queue = asyncio.Queue()
        full_text_future = asyncio.Future()
        
        # 2. Concurrently run LLM generation, text chunking, and TTS synthesis
        # ASYNC INTEGRITY: This gather block is the heart of the concurrent processing,
        # preserving the original script's low-latency streaming behavior.
        llm_task = asyncio.create_task(self.llm_engine.generate_response(transcribed_text, text_token_queue))
        chunker_task = asyncio.create_task(self.text_chunker(text_token_queue, tts_sentence_queue, full_text_future))
        tts_task = asyncio.create_task(self.tts_consumer(tts_sentence_queue))
        
        await asyncio.gather(llm_task, chunker_task, tts_task)
        return await full_text_future

    async def text_chunker(self, text_queue: asyncio.Queue, tts_queue: asyncio.Queue, full_text_future: asyncio.Future):
        """
        Consumes tokens from the LLM, assembles them into sentences, and passes
        them to the TTS queue.
        """
        sentence_buffer = ""
        full_response_text = ""
        sentence_terminators = {".", "?", "!"}
        while True:
            token = await text_queue.get()
            if token is None:  # End of stream sentinel
                if sentence_buffer.strip():
                    await tts_queue.put(sentence_buffer.strip())
                await tts_queue.put(None)  # Signal end to TTS consumer
                break
            
            print(token, end="", flush=True) # User feedback
            sentence_buffer += token
            full_response_text += token

            if any(term in token for term in sentence_terminators):
                await tts_queue.put(sentence_buffer.strip())
                sentence_buffer = ""
        
        full_text_future.set_result(full_response_text)

    async def tts_consumer(self, tts_queue: asyncio.Queue):
        """
        Consumes sentences from the TTS queue, synthesizes them into audio,
        and adds the audio data to the player's queue.
        """
        loop = asyncio.get_running_loop()
        while True:
            text_chunk = await tts_queue.get()
            if text_chunk is None:  # End of queue sentinel
                break
            
            # 3. Synchronous TTS in its dedicated executor
            audio_data = await loop.run_in_executor(
                self.services.tts_executor, self.tts_engine.synthesize_speech_sync, text_chunk
            )
            await self.services.player.add_to_queue(audio_data)