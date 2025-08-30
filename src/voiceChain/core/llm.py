import asyncio
import time
from loguru import logger

class LLMEngine:
    """
    Manages the lifecycle of a Llama-cpp model, including loading and asynchronous
    generation, while preserving performance characteristics.
    """
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 32768, verbose: bool = False):
        """
        Initializes the LLMEngine and pre-loads the language model into memory.
        This is a blocking operation that consumes significant memory and time.
        """
        self.llm = self._load_llm(model_path, n_gpu_layers, n_ctx, verbose)
        if self.llm:
            logger.success("LLMEngine initialized with pre-loaded model.")
        else:
            logger.critical("LLMEngine failed to initialize.")
    
    def _load_llm(self, model_path, n_gpu_layers, n_ctx, verbose):
        logger.info("Loading LLM... (This may take a moment)")
        try:
            from llama_cpp import Llama
            # PERFORMANCE: n_gpu_layers=-1 offloads all possible layers to the Metal GPU on Apple Silicon.
            llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=verbose)
            logger.success("LLM loaded into memory.")
            return llm
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load LLM. Agent cannot start. Error: {e}")
            return None

    async def generate_response(self, user_prompt: str, text_queue: asyncio.Queue):
        """
        Generates a streaming response from the LLM and puts tokens into an asyncio queue.
        This async method wraps the synchronous, blocking llama-cpp call in a thread executor.
        """
        if not self.llm:
            logger.error("LLM not loaded, cannot generate response.")
            await text_queue.put(None)
            return

        prompt_messages = [
            {"role": "system", "content": "You are a helpful, brief, and conversational AI assistant. Your responses must be broken into complete sentences, ending with proper punctuation. Do not use emojis."},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(f"Starting LLM stream for prompt: '{user_prompt}'")
        start_time = time.time()
        loop = asyncio.get_running_loop()
        
        # ASYNC INTEGRITY: Preserving the original pattern of running the blocking
        # generator in an executor to avoid stalling the event loop.
        try:
            stream = await loop.run_in_executor(
                None, 
                lambda: self.llm.create_chat_completion(
                    messages=prompt_messages, max_tokens=150, stream=True
                )
            )
            for chunk in stream:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    await text_queue.put(delta['content'])
        except Exception as e:
            logger.error(f"Error during LLM stream generation: {e}", exc_info=True)
        finally:
            await text_queue.put(None) # Sentinel value to signal end of stream
            logger.success(f"LLM stream finished in {time.time() - start_time:.2f}s.")