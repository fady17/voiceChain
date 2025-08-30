import asyncio
import numpy as np
from loguru import logger

from .services import ServiceManager
from .runner import PipelineRunner
from ..core.stt import Transcriber
from ..core.llm import LLMEngine
from ..core.tts import TextToSpeechEngine
from ..utils.state import AgentState

class ConversationManager:
    """
    The main state machine for the voice agent. Manages the conversation state
    (IDLE, PROCESSING, RESPONDING) and handles user barge-in.
    """
    def __init__(self, services: ServiceManager, transcriber: Transcriber, llm_engine: LLMEngine, tts_engine: TextToSpeechEngine):
        self.state = AgentState.IDLE
        self.services = services
        # The runner is composed here, using the injected dependencies.
        self.pipeline_runner = PipelineRunner(transcriber, llm_engine, tts_engine, services)
        self.active_pipeline_task = None
        self.current_agent_utterance = ""

    async def run(self):
        """
        The main execution loop for the conversation manager.
        """
        logger.info("Conversation Manager started. Listening for speech...")
        while True:
            user_audio_data = await self.services.user_utterance_queue.get()
            
            if self.state == AgentState.RESPONDING:
                is_barge_in = await self.check_for_barge_in(user_audio_data)
                if is_barge_in:
                    await self.handle_barge_in(user_audio_data)
            elif self.state == AgentState.IDLE:
                if self.active_pipeline_task and not self.active_pipeline_task.done():
                    logger.warning("User spoke while processing a previous turn. Ignoring.")
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
            # Wait for the audio player's queue to be empty
            while not self.services.player._playback_queue.empty():
                await asyncio.sleep(0.1)
            # Wait for the stream to report it's no longer active
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

    async def check_for_barge_in(self, audio_data: np.ndarray) -> bool:
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
        # Simple normalization to check if user speech is just an echo of the agent's speech
        agent_norm = ''.join(c for c in self.current_agent_utterance if c.isalnum()).lower()
        user_norm = ''.join(c for c in user_text if c.isalnum()).lower()
        if not user_norm:
            return True # Empty transcription is likely echo/noise
        if user_norm in agent_norm:
            logger.debug(f"Echo detected: '{user_text}' is part of agent response.")
            return True
        return False

    async def handle_barge_in(self, audio_data: np.ndarray):
        logger.warning("Handling barge-in...")
        # Cancel the currently running pipeline
        if self.active_pipeline_task:
            self.active_pipeline_task.cancel()
        
        # Interrupt audio playback
        await self.services.player.interrupt()
        
        # Reset executors to discard any pending tasks
        self.services.reset_executors()
        
        # Start a new turn with the interrupting audio
        await self.start_new_turn(audio_data)

    async def transition_to(self, new_state: AgentState):
        if self.state != new_state:
            logger.info(f"State transition: {self.state.name} -> {new_state.name}")
            self.state = new_state