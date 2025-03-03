"""Audio conversion service with proper streaming support"""

import struct
from io import BytesIO
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger
from pydub import AudioSegment
import av

class StreamingAudioWriter:
    """Handles streaming audio format conversions"""

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts=0

        codec_map = {"wav":"pcm_s16le","mp3":"mp3","opus":"libopus","flac":"flac", "aac":"aac"}
        # Format-specific setup
        if self.format in ["wav", "opus","flac","mp3","aac","pcm"]:
            if self.format != "pcm":
                self.output_buffer = BytesIO()
                self.container = av.open(self.output_buffer, mode="w", format=self.format)
                self.stream = self.container.add_stream(codec_map[self.format],sample_rate=self.sample_rate,layout='mono' if self.channels == 1 else 'stereo')
                self.stream.bit_rate = 128000
        else:
            raise ValueError(f"Unsupported format: {format}")

    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.

        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """

        if finalize:
            if self.format != "pcm":
                packets = self.stream.encode(None)
                for packet in packets:
                    self.container.mux(packet)
                    
                data = self.output_buffer.getvalue()
                self.container.close()
                
                # MP3格式特殊处理，修复格式问题
                if self.format == "mp3":
                    data = self._fix_mp3_format(data)
                
                return data

        if audio_data is None or len(audio_data) == 0:
            return b""

        if self.format == "pcm":
            # Write raw bytes
            return audio_data.tobytes()
        else:
            frame = av.AudioFrame.from_ndarray(audio_data.reshape(1, -1), format='s16', layout='mono' if self.channels == 1 else 'stereo')
            frame.sample_rate=self.sample_rate

            
            frame.pts = self.pts
            self.pts += frame.samples
            
            packets = self.stream.encode(frame)
            for packet in packets:
                self.container.mux(packet)
            
            data = self.output_buffer.getvalue()
            
            # MP3格式中间块不需要修复，只在最终输出修复
            
            self.output_buffer.seek(0)
            self.output_buffer.truncate(0)
            return data 
    
    def _fix_mp3_format(self, mp3_data: bytes) -> bytes:
        """修复MP3格式，确保符合标准格式
        
        Args:
            mp3_data: 原始MP3数据
            
        Returns:
            修复后的MP3数据
        """
        try:
            # 将二进制数据转换为BytesIO，然后使用pydub加载
            mp3_buffer = BytesIO(mp3_data)
            audio = AudioSegment.from_file(mp3_buffer, format="mp3")
            
            # 重新导出为MP3格式，这将重建正确的MP3头和文件结构
            # 指定比特率为128k，与原始设置保持一致
            output_buffer = BytesIO()
            audio.export(output_buffer, format="mp3", bitrate="128k")
            
            # 返回修复后的MP3数据
            return output_buffer.getvalue()
        except Exception as e:
            logger.error(f"Error fixing MP3 format: {str(e)}")
            # 如果修复失败，返回原始数据
            return mp3_data

