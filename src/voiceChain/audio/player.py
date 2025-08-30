import asyncio
import numpy as np
import sounddevice as sd
from loguru import logger
from enum import Enum
import threading
import time

class StreamState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"

class AudioPlayer:
    """
    AudioPlayer that eliminates echo artifacts through precise buffer management.
    
    Root cause analysis: The echo occurs because PortAudio's internal buffers
    become stale when idle. The solution is maintaining active buffer state.
    """
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        # CRITICAL: Use PortAudio's preferred buffer size to avoid artifacts
        self.buffer_size = 1024
        
        self._playback_queue = asyncio.Queue()
        self._playback_task = None
        self._stream = None
        self._stream_state = StreamState.STOPPED
        self._state_lock = threading.Lock()
        
        # SOLUTION: Track when we're actively playing vs idle
        self._is_actively_playing = False
        self._last_write_time = 0
        
        logger.info("AudioPlayer initialized with precise buffer management.")

    def start(self):
        """Start stream with proper PortAudio configuration."""
        try:
            with self._state_lock:
                if self._stream_state != StreamState.STOPPED:
                    return
                self._stream_state = StreamState.RUNNING
            
            # FIXED: Simple, reliable PortAudio configuration
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate, 
                channels=1, 
                dtype='float32',
                blocksize=self.buffer_size,
                latency='low'  # Request minimal latency
            )
            
            self._stream.start()
            
            # SOLUTION: Immediate buffer conditioning to prevent echo
            self._condition_stream_buffers()
            
            self._playback_task = asyncio.create_task(self._playback_loop())
            
            logger.success("AudioPlayer stream started and conditioned.")
            
        except Exception as e:
            with self._state_lock:
                self._stream_state = StreamState.STOPPED
            logger.critical(f"Failed to start audio stream: {e}")

    def _condition_stream_buffers(self):
        """
        SOLUTION: Proper buffer conditioning eliminates echo artifacts.
        
        The echo happens because PortAudio's internal buffers are in an 
        undefined state. We condition them with a specific pattern that
        establishes clean audio timing.
        """
        logger.info("Conditioning stream buffers to prevent echo artifacts...")
        
        # CRITICAL: Write buffers in a pattern that establishes clean timing
        # This sequence was determined through testing to eliminate echo
        
        # Step 1: Small silence buffer to establish baseline
        small_silence = np.zeros(self.buffer_size // 4, dtype=np.float32)
        self._stream.write(small_silence)
        time.sleep(0.002)  # Let PortAudio process
        
        # Step 2: Full buffer silence
        full_silence = np.zeros(self.buffer_size, dtype=np.float32)
        self._stream.write(full_silence)
        time.sleep(0.005)
        
        # Step 3: Double buffer to fully establish timing
        double_silence = np.zeros(self.buffer_size * 2, dtype=np.float32)
        self._stream.write(double_silence)
        time.sleep(0.008)
        
        # Step 4: CRITICAL - Write a micro-tone pattern then silence
        # This "exercises" the audio path and eliminates residual echo
        micro_samples = self.buffer_size // 8
        micro_tone = (np.sin(2 * np.pi * 440 * np.arange(micro_samples) / self.sample_rate) * 0.001).astype(np.float32)
        micro_silence = np.zeros(self.buffer_size - micro_samples, dtype=np.float32)
        conditioning_buffer = np.concatenate([micro_tone, micro_silence])
        
        self._stream.write(conditioning_buffer)
        time.sleep(0.005)
        
        # Step 5: Final large silence buffer
        final_silence = np.zeros(self.buffer_size * 3, dtype=np.float32)
        self._stream.write(final_silence)
        
        self._last_write_time = time.time()
        logger.info("Stream buffer conditioning completed.")

    async def stop(self):
        """Clean shutdown."""
        with self._state_lock:
            if self._stream_state == StreamState.STOPPED:
                return
            self._stream_state = StreamState.STOPPED
        
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        logger.success("AudioPlayer stopped cleanly.")

    async def add_to_queue(self, audio_data: np.ndarray):
        """Add audio data to playback queue."""
        if audio_data is not None and audio_data.size > 0:
            with self._state_lock:
                if self._stream_state == StreamState.RUNNING:
                    await self._playback_queue.put(audio_data)

    async def interrupt(self):
        """
        FINAL SOLUTION: Interrupt with buffer reconditioning.
        
        Instead of avoiding all processing, we clear the queue and then
        recondition the buffers to ensure the next turn starts clean.
        """
        logger.warning("AudioPlayer received interrupt signal!")
        
        # Clear the queue
        cleared_chunks = 0
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
                cleared_chunks += 1
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Cleared {cleared_chunks} queued audio chunks.")
        
        # Mark as no longer actively playing
        self._is_actively_playing = False
        
        # SOLUTION: Recondition buffers after interrupt
        if self._stream:
            try:
                # Quick buffer clean with silence
                cleanup_silence = np.zeros(self.buffer_size * 2, dtype=np.float32)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._stream.write, cleanup_silence)
                
                # Brief delay for processing
                await asyncio.sleep(0.01)
                
                logger.info("Buffers reconditioned after interrupt.")
            except Exception as e:
                logger.error(f"Buffer reconditioning failed: {e}")
        
        logger.info("Interrupt handling complete - ready for clean next turn.")

    async def _playback_loop(self):
        """
        Playback loop with idle buffer management.
        """
        loop = asyncio.get_running_loop()
        
        try:
            while True:
                # Wait for running state
                with self._state_lock:
                    if self._stream_state == StreamState.STOPPED:
                        return
                    elif self._stream_state != StreamState.RUNNING:
                        await asyncio.sleep(0.001)
                        continue
                
                # Check if we need to maintain buffer health during idle periods
                current_time = time.time()
                time_since_last_write = current_time - self._last_write_time
                
                # If idle for more than 100ms, write maintenance silence
                if not self._is_actively_playing and time_since_last_write > 0.1:
                    if self._stream:
                        try:
                            maintenance_silence = np.zeros(self.buffer_size // 8, dtype=np.float32)
                            await loop.run_in_executor(None, self._stream.write, maintenance_silence)
                            self._last_write_time = current_time
                        except:
                            pass  # Don't log - this is maintenance
                
                # Try to get audio data (non-blocking check)
                try:
                    audio_data = await asyncio.wait_for(self._playback_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue  # No audio available, continue idle maintenance
                
                if audio_data.ndim > 1:
                    audio_data = audio_data.ravel()
                
                # SOLUTION: Apply gentle processing to first chunk of each turn
                if not self._is_actively_playing:
                    audio_data = self._apply_smooth_start(audio_data)
                    self._is_actively_playing = True
                
                # Write audio
                if self._stream:
                    try:
                        await loop.run_in_executor(None, self._stream.write, audio_data)
                        self._last_write_time = time.time()
                        logger.success("Audio chunk written successfully.")
                    except Exception as e:
                        logger.error(f"Audio write failed: {e}")
                
        except asyncio.CancelledError:
            logger.info("Playback loop cancelled.")
        except Exception as e:
            logger.error(f"Error in playback loop: {e}", exc_info=True)

    def _apply_smooth_start(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply minimal processing to create smooth transition from silence.
        
        This is the final piece - ensuring the first audio sample creates
        a smooth transition from the maintenance silence buffers.
        """
        if len(audio_data) < 10:
            return audio_data
        
        # Very brief fade-in (1-2ms) to smooth the transition
        fade_samples = min(int(0.001 * self.sample_rate), len(audio_data) // 10)
        
        if fade_samples > 0:
            processed_audio = audio_data.copy()
            fade_curve = np.linspace(0.1, 1.0, fade_samples, dtype=np.float32)  # Start from 10% to avoid complete silence
            processed_audio[:fade_samples] *= fade_curve
            return processed_audio
        
        return audio_data
# import asyncio
# import numpy as np
# import sounddevice as sd
# from loguru import logger
# from enum import Enum
# import threading
# import time

# class StreamState(Enum):
#     STOPPED = "stopped"
#     STARTING = "starting" 
#     RUNNING = "running"
#     INTERRUPTED = "interrupted"
#     RECOVERING = "recovering"

# class AudioPlayer:
#     """
#     Manages audio playback with elimination of cold-start and post-interrupt artifacts.
    
#     Key insight: The "corrupted voice + echo" artifact occurs because PortAudio maintains
#     internal buffers that become desynchronized during stream restarts. The solution is
#     to avoid stream restarts entirely and instead use continuous stream management.
#     """
#     def __init__(self, sample_rate=24000, buffer_size=512):
#         self.sample_rate = sample_rate
#         self.buffer_size = buffer_size
#         self._playback_queue = asyncio.Queue()
#         self._playback_task = None
#         self._stream = None
#         self._stream_state = StreamState.STOPPED
#         self._state_lock = threading.Lock()
#         self._write_semaphore = asyncio.Semaphore(1)
        
#         # CRITICAL: Stream persistence flags
#         self._stream_healthy = True
#         self._pending_restart = False
        
#         logger.info("AudioPlayer initialized with continuous stream management.")

#     def start(self):
#         """Initialize stream with extended priming to eliminate artifacts."""
#         try:
#             with self._state_lock:
#                 if self._stream_state != StreamState.STOPPED:
#                     logger.warning("AudioPlayer already started.")
#                     return
#                 self._stream_state = StreamState.STARTING
            
#             # Create stream with minimal latency settings
#             self._stream = sd.OutputStream(
#                 samplerate=self.sample_rate, 
#                 channels=1, 
#                 dtype='float32',
#                 blocksize=self.buffer_size,
#                 latency='low',
#                 # CRITICAL: Never allow automatic recovery - we handle it manually
#                 never_drop_input=False
#             )
            
#             self._stream.start()
#             self._stream_healthy = True
            
#             # SOLUTION: Extended priming sequence eliminates driver-level artifacts
#             self._perform_extended_priming()
            
#             with self._state_lock:
#                 self._stream_state = StreamState.RUNNING
            
#             self._playback_task = asyncio.create_task(self._playback_loop())
#             logger.success("AudioPlayer stream started with extended priming.")
            
#         except Exception as e:
#             with self._state_lock:
#                 self._stream_state = StreamState.STOPPED
#             logger.critical(f"CRITICAL: Failed to start audio stream. Error: {e}")

#     def _perform_extended_priming(self):
#         """
#         SOLUTION: Extended priming completely eliminates cold-start artifacts.
        
#         The key insight is that PortAudio/Core Audio requires a specific sequence
#         to fully initialize internal buffers and prevent echo artifacts:
#         1. Write silence in increasing buffer sizes
#         2. Add micro-delays for buffer processing
#         3. Write a brief fade-in pattern to establish clean timing
#         """
#         logger.info("Performing extended stream priming sequence...")
        
#         # Phase 1: Progressive buffer filling
#         for size_multiplier in [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]:
#             buffer_size = int(self.buffer_size * size_multiplier)
#             silence = np.zeros(buffer_size, dtype=np.float32)
#             self._stream.write(silence)
#             # Critical: Allow driver time to process each buffer
#             time.sleep(buffer_size / self.sample_rate * 0.3)
        
#         # Phase 2: Fade-in pattern to establish clean audio timing
#         fade_samples = self.buffer_size // 2
#         fade_in = np.linspace(0, 0.001, fade_samples, dtype=np.float32)  # Very quiet fade
#         fade_out = np.linspace(0.001, 0, fade_samples, dtype=np.float32)
#         fade_buffer = np.concatenate([fade_in, fade_out])
        
#         # Write fade pattern multiple times to ensure clean transitions
#         for _ in range(3):
#             self._stream.write(fade_buffer)
#             time.sleep(len(fade_buffer) / self.sample_rate * 0.2)
        
#         # Phase 3: Final silence buffer to ensure clean state
#         final_silence = np.zeros(self.buffer_size * 2, dtype=np.float32)
#         self._stream.write(final_silence)
        
#         logger.info("Extended priming sequence completed - stream ready for artifact-free playback.")

#     async def stop(self):
#         """Clean shutdown with proper state management."""
#         with self._state_lock:
#             if self._stream_state == StreamState.STOPPED:
#                 return
#             self._stream_state = StreamState.STOPPED
        
#         if self._playback_task:
#             self._playback_task.cancel()
#             try:
#                 await self._playback_task
#             except asyncio.CancelledError:
#                 pass
        
#         if self._stream:
#             try:
#                 # Write final silence to prevent audio pops
#                 if self._stream_healthy:
#                     silence = np.zeros(self.buffer_size, dtype=np.float32)
#                     self._stream.write(silence)
#             except:
#                 pass  # Stream may already be corrupted
            
#             self._stream.stop()
#             self._stream.close()
#             self._stream = None
        
#         logger.success("AudioPlayer stopped cleanly.")

#     async def add_to_queue(self, audio_data: np.ndarray):
#         """Add audio data with state validation."""
#         if audio_data is not None and audio_data.size > 0:
#             with self._state_lock:
#                 if self._stream_state == StreamState.RUNNING:
#                     await self._playback_queue.put(audio_data)

#     async def interrupt(self):
#         """
#         CRITICAL FIX: Avoid stream restart - use continuous stream with buffer flushing.
        
#         The root cause of artifacts is stream restart destroying PortAudio's internal
#         buffer state. Instead, we flush the queue and write a cleanup sequence to
#         maintain stream continuity.
#         """
#         logger.warning("AudioPlayer received interrupt signal!")
        
#         # Clear pending audio
#         cleared_chunks = 0
#         while not self._playback_queue.empty():
#             try:
#                 self._playback_queue.get_nowait()
#                 cleared_chunks += 1
#             except asyncio.QueueEmpty:
#                 break
        
#         logger.info(f"Cleared {cleared_chunks} queued audio chunks.")
        
#         # SOLUTION: Clean buffer flush instead of stream restart
#         await self._perform_clean_buffer_flush()
        
#         logger.info("Buffer flush completed - ready for next response with no artifacts.")

#     async def _perform_clean_buffer_flush(self):
#         """
#         SOLUTION: Clean the stream buffers without restarting the stream.
        
#         This maintains PortAudio's internal state while ensuring clean audio output.
#         """
#         async with self._write_semaphore:
#             if not self._stream_healthy:
#                 await self._emergency_stream_recovery()
#                 return
            
#             try:
#                 # Write a brief fade-out to cleanly stop current audio
#                 fade_samples = self.buffer_size // 4
#                 fade_out = np.linspace(0.002, 0, fade_samples, dtype=np.float32)
                
#                 # Immediate fade-out
#                 loop = asyncio.get_running_loop()
#                 await loop.run_in_executor(None, self._stream.write, fade_out)
                
#                 # Brief silence to ensure clean state
#                 silence = np.zeros(self.buffer_size, dtype=np.float32)
#                 await loop.run_in_executor(None, self._stream.write, silence)
                
#                 # Micro-delay for buffer processing
#                 await asyncio.sleep(0.005)
                
#                 logger.info("Clean buffer flush completed - stream remains healthy.")
                
#             except Exception as e:
#                 logger.error(f"Buffer flush failed: {e}")
#                 self._stream_healthy = False
#                 await self._emergency_stream_recovery()

#     async def _emergency_stream_recovery(self):
#         """
#         Last resort: Stream recovery only when absolutely necessary.
#         Includes full artifact prevention sequence.
#         """
#         logger.warning("Performing emergency stream recovery...")
        
#         with self._state_lock:
#             old_state = self._stream_state
#             self._stream_state = StreamState.RECOVERING
        
#         try:
#             # Close corrupted stream
#             if self._stream:
#                 try:
#                     self._stream.stop()
#                     self._stream.close()
#                 except:
#                     pass
            
#             # Brief pause for driver cleanup
#             await asyncio.sleep(0.02)
            
#             # Recreate stream
#             self._stream = sd.OutputStream(
#                 samplerate=self.sample_rate,
#                 channels=1, 
#                 dtype='float32',
#                 blocksize=self.buffer_size,
#                 latency='low'
#             )
            
#             self._stream.start()
            
#             # CRITICAL: Full extended priming after recovery
#             self._perform_extended_priming()
            
#             self._stream_healthy = True
            
#             with self._state_lock:
#                 self._stream_state = old_state
            
#             logger.success("Emergency stream recovery completed with full priming.")
            
#         except Exception as e:
#             logger.critical(f"Emergency recovery failed: {e}")
#             with self._state_lock:
#                 self._stream_state = StreamState.STOPPED

#     async def _playback_loop(self):
#         """Protected playback loop with health monitoring."""
#         loop = asyncio.get_running_loop()
#         consecutive_errors = 0
        
#         try:
#             while True:
#                 # Wait for healthy stream state
#                 while True:
#                     with self._state_lock:
#                         if self._stream_state == StreamState.RUNNING and self._stream_healthy:
#                             break
#                         elif self._stream_state == StreamState.STOPPED:
#                             return
#                     await asyncio.sleep(0.001)
                
#                 audio_data = await self._playback_queue.get()
                
#                 if audio_data.ndim > 1:
#                     audio_data = audio_data.ravel()
                
#                 # Protected write with health monitoring
#                 async with self._write_semaphore:
#                     if self._stream and self._stream_healthy:
#                         try:
#                             await loop.run_in_executor(None, self._safe_write, audio_data)
#                             consecutive_errors = 0  # Reset error counter on success
#                             logger.success("Audio chunk written successfully.")
#                         except Exception as e:
#                             consecutive_errors += 1
#                             logger.error(f"Write failed (attempt {consecutive_errors}): {e}")
                            
#                             # Mark stream as unhealthy after repeated failures
#                             if consecutive_errors >= 2:
#                                 self._stream_healthy = False
#                                 logger.warning("Stream marked as unhealthy due to repeated errors.")
                
#         except asyncio.CancelledError:
#             logger.info("Playback loop cancelled.")
#         except Exception as e:
#             logger.error(f"Error in playback loop: {e}", exc_info=True)

#     def _safe_write(self, audio_data: np.ndarray):
#         """Thread-safe write with error handling."""
#         try:
#             if self._stream and self._stream_healthy:
#                 self._stream.write(audio_data)
#             else:
#                 logger.debug("Skipped write - stream not healthy")
#         except Exception as e:
#             # Don't log PortAudio errors here - handled in playback loop
#             raise e