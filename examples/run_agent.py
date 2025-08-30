import sys
import os
import asyncio
from loguru import logger

# --- Start of Path Modification (Corrected for 'src' layout) ---
# This block adds the project's 'src' directory to the Python path.
# This is necessary to allow the script to find the 'voiceChain' module.

# Get the absolute path to the project's root directory
# __file__ -> .../voice/examples/run_agent.py
# dirname -> .../voice/examples
# dirname -> .../voice
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The voiceChain package is inside the 'src' directory.
# We must add '.../voice/src' to the path.
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
# --- End of Path Modification ---

# Now that '.../voice/src' is in the path, Python can find the 'voiceChain' package.
from voiceChain.utils.logging import setup_logging
from voiceChain.core.stt import Transcriber
from voiceChain.core.tts import TextToSpeechEngine
from voiceChain.core.llm import LLMEngine
from voiceChain.pipeline.services import ServiceManager
from voiceChain.pipeline.manager import ConversationManager

async def main():
    """
    The main entry point for the voice agent application.
    This function acts as the "composition root", where all the library
    components are instantiated and wired together.
    """
    setup_logging(level="INFO")

    loop = asyncio.get_running_loop()

    # --- 1. Instantiate Core AI Engines ---
    logger.info("Pre-loading all AI models...")
    # NOTE: The model paths are relative to where you RUN the script.
    # Since we run from the root ('voice %'), the path 'models/...' is correct.
    transcriber = Transcriber(model_path="models/whisper-large-v3-turbo")
    tts_engine = TextToSpeechEngine(model_path="models/Kokoro", speaker="af_heart")
    llm_engine = LLMEngine(model_path="./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", n_gpu_layers=-1)
    logger.success("All AI models loaded.")

    # --- 2. Instantiate Services ---
    services = ServiceManager(loop)

    # --- 3. Instantiate and Run the Conversation Manager ---
    manager = ConversationManager(
        services=services,
        transcriber=transcriber,
        llm_engine=llm_engine,
        tts_engine=tts_engine
    )

    try:
        await services.start()
        await manager.run()
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    finally:
        logger.info("Shutting down agent...")
        await services.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")