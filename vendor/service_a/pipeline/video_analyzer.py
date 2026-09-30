from dataclasses import dataclass
from fractions import Fraction
from pipeline.ffmpeg_utils import probe

@dataclass
class VideoMetadata:
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    audio_streams: list
    subtitle_streams: list
    sample_aspect_ratio: str

def analyze(path, config):
    data = probe(path, config)
    streams = data.get('streams', [])
    video = next((s for s in streams if s['codec_type']=='video' and not s.get('disposition',{}).get('attached_pic')),None)
    if not video: raise ValueError('No video stream found')
    rate = video.get('avg_frame_rate','0/1')
    fps = float(Fraction(rate)) if rate != '0/0' else 0
    duration = float(data.get('format',{}).get('duration',video.get('duration',0)))
    if duration <= 0: raise ValueError('Video duration is missing or invalid')
    return VideoMetadata(duration,video['width'],video['height'],fps,video['codec_name'],
                         [s for s in streams if s['codec_type']=='audio'],
                         [s for s in streams if s['codec_type']=='subtitle'],video.get('sample_aspect_ratio','1:1'))
