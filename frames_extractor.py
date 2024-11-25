import ffmpeg
import subprocess
from pathlib import Path
from typing import Optional, Union
from logging import Logger, getLogger
from dataclasses import dataclass


class FFmpegError(Exception):
    """Custom exception for FFmpeg-related errors."""
    pass


@dataclass
class ExtractorConfig:
    """Configuration class for frame extraction settings."""
    video_path: Union[str, Path]
    execution_number: int
    fps: float
    output_quality: int = 2  # Quality scale for JPEG (2-31, lower is better)
    video_bitrate: str = '5000k'


class FrameExtractor:
    def __init__(self, config: ExtractorConfig, logger: Optional[Logger] = None):
        """
        Initialize the FrameExtractor.

        Args:
            config: ExtractorConfig object containing extraction parameters
            logger: Optional logger instance. If None, creates a new logger.
        """
        self.config = config
        self.logger = logger or getLogger(__name__)
        self.output_dir = self._create_output_directory()
        self.total_frames = 0  # Initialize total_frames
        self.input_fps = config.fps  # FPS do vídeo de entrada
        self.frame_name_template = 'frame%06d.png'  # Template de nome com zeros à esquerda

    def _create_output_directory(self) -> Path:
        """
        Create and return the output directory path for frames.

        Returns:
            Path object pointing to the created directory
        """
        output_dir = Path("Outputs") / f"Exec_{self.config.execution_number}" / "Frames_extracted"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def _get_video_info(self) -> Optional[dict]:
        """
        Get video information using ffmpeg.

        Returns:
            Dictionary containing video stream information or None if probe fails

        Raises:
            FFmpegError: If ffmpeg probe fails
        """
        try:
            probe = ffmpeg.probe(str(self.config.video_path))
            video_info = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            if not video_info:
                self.logger.error("No video stream found in file")
                return None

            self.logger.debug(f"Video info retrieved: {video_info}")

            # Obter o número total de frames
            nb_frames = video_info.get('nb_frames')
            if nb_frames is not None and nb_frames.isdigit():
                self.total_frames = int(nb_frames)
            else:
                # Se 'nb_frames' não estiver disponível, calcular a partir da duração e frame rate
                duration = float(video_info.get('duration', 0))
                frame_rate_str = video_info['avg_frame_rate']
                frame_rate = self._parse_frame_rate(frame_rate_str)
                self.total_frames = int(duration * frame_rate)

            return video_info

        except subprocess.CalledProcessError as e:
            error_message = f"Error reading video information: {e.stderr.decode() if e.stderr else str(e)}"
            self.logger.error(error_message)
            raise FFmpegError(error_message)
        except Exception as e:
            error_message = f"Unexpected error during video probe: {str(e)}"
            self.logger.error(error_message)
            raise FFmpegError(error_message)

    def _parse_frame_rate(self, frame_rate_str: str) -> float:
        """
        Parse frame rate string from ffmpeg into a float.

        Args:
            frame_rate_str: Frame rate string in format 'num/den' or 'num'

        Returns:
            Float representation of frame rate

        Raises:
            ValueError: If frame rate string cannot be parsed
        """
        try:
            if '/' in frame_rate_str:
                num, den = map(float, frame_rate_str.split('/'))
                return num / den
            return float(frame_rate_str)
        except (ValueError, ZeroDivisionError) as e:
            self.logger.error(f"Error parsing frame rate '{frame_rate_str}': {str(e)}")
            raise ValueError(f"Invalid frame rate format: {frame_rate_str}")

    def extract_frames(self) -> int:
        """
        Extract frames from video using ffmpeg.

        Returns:
            Number of frames extracted

        Raises:
            FFmpegError: If frame extraction fails
            ValueError: If video information cannot be retrieved
        """
        try:
            video_info = self._get_video_info()
            if not video_info:
                raise ValueError("Could not get video information")

            # Calculate the frame extraction interval
            input_fps = self._parse_frame_rate(video_info['r_frame_rate'])
            self.logger.info(f"Input video FPS: {input_fps}, Target FPS: {self.config.fps}")

            # Setup ffmpeg stream
            stream = ffmpeg.input(str(self.config.video_path))
            stream = ffmpeg.filter(stream, 'fps', fps=self.config.fps)

            output_path = str(self.output_dir / self.frame_name_template)
            stream = ffmpeg.output(
                stream,
                output_path,
                video_bitrate=self.config.video_bitrate,
                **{'qscale:v': self.config.output_quality}
            )

            # Run the ffmpeg command
            self.logger.info("Starting frame extraction...")
            try:
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            except subprocess.CalledProcessError as e:
                error_message = f"FFmpeg execution failed: {e.stderr.decode() if e.stderr else str(e)}"
                self.logger.error(error_message)
                raise FFmpegError(error_message)

            # Count extracted frames
            frame_count = len(list(self.output_dir.glob('frame*.png')))
            self.logger.info(f"Successfully extracted {frame_count} frames to {self.output_dir}")
            return frame_count

        except subprocess.CalledProcessError as e:
            error_message = f"FFmpeg process error: {e.stderr.decode() if e.stderr else str(e)}"
            self.logger.error(error_message)
            raise FFmpegError(error_message)
        except Exception as e:
            error_message = f"Unexpected error during frame extraction: {str(e)}"
            self.logger.error(error_message)
            raise FFmpegError(error_message)
