"""Clean Kokoro implementation with controlled resource management."""

import os
import gc
from contextlib import contextmanager
from typing import AsyncGenerator, Dict, Optional, Tuple, Union

import numpy as np
import torch
from kokoro import KModel, KPipeline
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import model_config
from ..structures.schemas import WordTimestamp
from .base import AudioChunk, BaseModelBackend


@contextmanager
def memory_cleanup_context(device: str):
    """Context manager for automatic memory cleanup after operations."""
    try:
        yield
    finally:
        # Force cleanup regardless of device
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()


def clean_temp_voice_files(voice_path: str):
    """Clean up temporary voice files."""
    try:
        if voice_path and os.path.exists(voice_path) and "temp_voice_" in os.path.basename(voice_path):
            os.remove(voice_path)
            logger.debug(f"Cleaned up temp voice file: {voice_path}")
    except Exception as e:
        logger.warning(f"Failed to clean temp voice file {voice_path}: {e}")


class KokoroV1(BaseModelBackend):
    """Kokoro backend with controlled resource management."""

    def __init__(self):
        """Initialize backend with environment-based configuration."""
        super().__init__()
        # Strictly respect settings.use_gpu
        self._device = settings.get_device()
        self._model: Optional[KModel] = None
        self._pipelines: Dict[str, KPipeline] = {}  # Store pipelines by lang_code
        self._pipeline_use_count: Dict[str, int] = {}  # Track pipeline usage for cleanup
        self._generation_count = 0  # Track total generations for periodic cleanup

    async def load_model(self, path: str) -> None:
        """Load pre-baked model.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

            if not os.path.exists(config_path):
                raise RuntimeError(f"Config file not found: {config_path}")

            logger.info(f"Loading Kokoro model on {self._device}")
            logger.info(f"Config path: {config_path}")
            logger.info(f"Model path: {model_path}")

            # Load model and let KModel handle device mapping
            self._model = KModel(config=config_path, model=model_path).eval()
            # For MPS, manually move ISTFT layers to CPU while keeping rest on MPS
            if self._device == "mps":
                logger.info(
                    "Moving model to MPS device with CPU fallback for unsupported operations"
                )
                self._model = self._model.to(torch.device("mps"))
            elif self._device == "cuda":
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        """Get or create pipeline for language code.

        Args:
            lang_code: Language code to use

        Returns:
            KPipeline instance for the language
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        if lang_code not in self._pipelines:
            # Clean up old pipelines if we have too many
            if len(self._pipelines) >= 5:  # Increase limit to 5 for better performance
                self._cleanup_old_pipelines()
            
            logger.info(f"Creating new pipeline for language code: {lang_code}")
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code, model=self._model, device=self._device
            )
            self._pipeline_use_count[lang_code] = 0
        
        self._pipeline_use_count[lang_code] += 1
        return self._pipelines[lang_code]

    def _cleanup_old_pipelines(self):
        """Clean up least used pipelines to free memory."""
        if len(self._pipelines) <= 2:  # Keep at least 2 pipelines
            return
            
        # Sort by usage count and remove least used
        sorted_pipelines = sorted(self._pipeline_use_count.items(), key=lambda x: x[1])
        lang_to_remove = sorted_pipelines[0][0]
        
        if lang_to_remove in self._pipelines:
            del self._pipelines[lang_to_remove]
            del self._pipeline_use_count[lang_to_remove]
            logger.debug(f"Cleaned up unused pipeline for lang_code: {lang_to_remove}")
            
            # Only clear memory after pipeline removal, not after every operation
            if self._device == "cuda":
                torch.cuda.empty_cache()

    def _should_cleanup_memory(self) -> bool:
        """Check if we should perform memory cleanup based on generation count and memory usage."""
        self._generation_count += 1
        
        # Only check memory every 10 generations to reduce overhead
        if self._generation_count % 10 != 0:
            return False
            
        if self._device == "cuda":
            memory_gb = torch.cuda.memory_allocated() / 1e9
            # Use original threshold, not reduced
            threshold = model_config.pytorch_gpu.memory_threshold
            return memory_gb > threshold
        
        return False

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio from phoneme tokens.

        Args:
            tokens: Input phoneme tokens to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        temp_voice_path = None
        try:
            # Only use memory cleanup context for CUDA and only if memory is high
            if self._device == "cuda" and self._should_cleanup_memory():
                context_manager = memory_cleanup_context(self._device)
            else:
                from contextlib import nullcontext
                context_manager = nullcontext()
                
            with context_manager:
                # Handle voice input
                voice_path: str
                voice_name: str
                if isinstance(voice, tuple):
                    voice_name, voice_data = voice
                    if isinstance(voice_data, str):
                        voice_path = voice_data
                    else:
                        # Save tensor to temporary file
                        import tempfile

                        temp_dir = tempfile.gettempdir()
                        voice_path = os.path.join(temp_dir, f"{voice_name}.pt")
                        # Save tensor with CPU mapping for portability
                        torch.save(voice_data.cpu(), voice_path)
                        temp_voice_path = voice_path  # Track for cleanup
                else:
                    voice_path = voice
                    voice_name = os.path.splitext(os.path.basename(voice_path))[0]

                # Load voice tensor with proper device mapping
                voice_tensor = await paths.load_voice_tensor(
                    voice_path, device=self._device
                )
                # Save back to a temporary file with proper device mapping
                import tempfile

                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(
                    temp_dir, f"temp_voice_{os.path.basename(voice_path)}"
                )
                await paths.save_voice_tensor(voice_tensor, temp_path)
                
                # Clean up the voice tensor from memory immediately
                del voice_tensor
                
                voice_path = temp_path
                temp_voice_path = temp_path  # Track for cleanup

                # Use provided lang_code, settings voice code override, or first letter of voice name
                if lang_code:  # api is given priority
                    pipeline_lang_code = lang_code
                elif settings.default_voice_code:  # settings is next priority
                    pipeline_lang_code = settings.default_voice_code
                else:  # voice name is default/fallback
                    pipeline_lang_code = voice_name[0].lower()

                pipeline = self._get_pipeline(pipeline_lang_code)

                logger.debug(
                    f"Generating audio from tokens with lang_code '{pipeline_lang_code}': '{tokens[:100]}{'...' if len(tokens) > 100 else ''}'"
                )
                
                for result in pipeline.generate_from_tokens(
                    tokens=tokens, voice=voice_path, speed=speed, model=self._model
                ):
                    if result.audio is not None:
                        logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                        # Convert to numpy and yield
                        audio_numpy = result.audio.numpy()
                        yield audio_numpy
                        
                        # Clean up the result tensor immediately
                        del result.audio
                    else:
                        logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                # Retry generation
                async for chunk in self.generate_from_tokens(
                    tokens, voice, speed, lang_code
                ):
                    yield chunk
            raise
        finally:
            # Always clean up temporary voice files
            if temp_voice_path:
                clean_temp_voice_files(temp_voice_path)

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio using model.

        Args:
            text: Input text to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
            
        temp_voice_path = None
        try:
            # Only use memory cleanup context for CUDA and only if memory is high
            if self._device == "cuda" and self._should_cleanup_memory():
                context_manager = memory_cleanup_context(self._device)
            else:
                from contextlib import nullcontext
                context_manager = nullcontext()
                
            with context_manager:
                # Handle voice input
                voice_path: str
                voice_name: str
                if isinstance(voice, tuple):
                    voice_name, voice_data = voice
                    if isinstance(voice_data, str):
                        voice_path = voice_data
                    else:
                        # Save tensor to temporary file
                        import tempfile

                        temp_dir = tempfile.gettempdir()
                        voice_path = os.path.join(temp_dir, f"{voice_name}.pt")
                        # Save tensor with CPU mapping for portability
                        torch.save(voice_data.cpu(), voice_path)
                        temp_voice_path = voice_path  # Track for cleanup
                else:
                    voice_path = voice
                    voice_name = os.path.splitext(os.path.basename(voice_path))[0]

                # Load voice tensor with proper device mapping
                voice_tensor = await paths.load_voice_tensor(
                    voice_path, device=self._device
                )
                # Save back to a temporary file with proper device mapping
                import tempfile

                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(
                    temp_dir, f"temp_voice_{os.path.basename(voice_path)}"
                )
                await paths.save_voice_tensor(voice_tensor, temp_path)
                
                # Clean up the voice tensor from memory immediately
                del voice_tensor
                
                voice_path = temp_path
                temp_voice_path = temp_path  # Track for cleanup

                # Use provided lang_code, settings voice code override, or first letter of voice name
                pipeline_lang_code = (
                    lang_code
                    if lang_code
                    else (
                        settings.default_voice_code
                        if settings.default_voice_code
                        else voice_name[0].lower()
                    )
                )
                pipeline = self._get_pipeline(pipeline_lang_code)

                logger.debug(
                    f"Generating audio for text with lang_code '{pipeline_lang_code}': '{text[:100]}{'...' if len(text) > 100 else ''}'"
                )
                
                for result in pipeline(
                    text, voice=voice_path, speed=speed, model=self._model
                ):
                    if result.audio is not None:
                        logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                        word_timestamps = None
                        if (
                            return_timestamps
                            and hasattr(result, "tokens")
                            and result.tokens
                        ):
                            word_timestamps = []
                            current_offset = 0.0
                            logger.debug(
                                f"Processing chunk timestamps with {len(result.tokens)} tokens"
                            )
                            if result.pred_dur is not None:
                                try:
                                    # Add timestamps with offset
                                    for token in result.tokens:
                                        if not all(
                                            hasattr(token, attr)
                                            for attr in [
                                                "text",
                                                "start_ts",
                                                "end_ts",
                                            ]
                                        ):
                                            continue
                                        if not token.text or not token.text.strip():
                                            continue

                                        start_time = float(token.start_ts) + current_offset
                                        end_time = float(token.end_ts) + current_offset
                                        word_timestamps.append(
                                            WordTimestamp(
                                                word=str(token.text).strip(),
                                                start_time=start_time,
                                                end_time=end_time,
                                            )
                                        )
                                        logger.debug(
                                            f"Added timestamp for word '{token.text}': {start_time:.3f}s - {end_time:.3f}s"
                                        )

                                except Exception as e:
                                    logger.error(
                                        f"Failed to process timestamps for chunk: {e}"
                                    )

                        # Convert to numpy and create chunk
                        audio_numpy = result.audio.numpy()
                        audio_chunk = AudioChunk(audio_numpy, word_timestamps=word_timestamps)
                        yield audio_chunk
                        
                        # Clean up the result tensor immediately
                        del result.audio
                    else:
                        logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                # Retry generation
                async for chunk in self.generate(text, voice, speed, lang_code):
                    yield chunk
            raise
        finally:
            # Always clean up temporary voice files
            if temp_voice_path:
                clean_temp_voice_files(temp_voice_path)

    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold."""
        if self._device == "cuda":
            memory_gb = torch.cuda.memory_allocated() / 1e9
            # Use original threshold
            threshold = model_config.pytorch_gpu.memory_threshold
            return memory_gb > threshold
        # MPS doesn't provide memory management APIs
        return False

    def _clear_memory(self) -> None:
        """Clear device memory aggressively."""
        logger.debug(f"Clearing memory for device: {self._device}")
        
        # Force garbage collection first
        gc.collect()
        
        if self._device == "cuda":
            # Get memory info before cleanup
            memory_before = torch.cuda.memory_allocated() / 1e9
            
            # Clear cache and synchronize
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get memory info after cleanup  
            memory_after = torch.cuda.memory_allocated() / 1e9
            logger.debug(f"GPU memory: {memory_before:.2f}GB -> {memory_after:.2f}GB")
            
        elif self._device == "mps":
            # Empty cache if available (future-proofing)
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def unload(self) -> None:
        """Unload model and free resources."""
        logger.info("Unloading Kokoro V1 model and cleaning up resources")
        
        # Clear pipelines first
        for lang_code, pipeline in self._pipelines.items():
            try:
                del pipeline
            except:
                pass
        self._pipelines.clear()
        self._pipeline_use_count.clear()
        
        # Clear model
        if self._model is not None:
            del self._model
            self._model = None
        
        # Force memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device
