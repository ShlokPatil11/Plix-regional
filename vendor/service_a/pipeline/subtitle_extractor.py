import re
import unicodedata
import pysubs2
from models.subtitle import SubtitleSegment
from pipeline.ffmpeg_utils import run

def normalize(text):
    lines = []
    for line in unicodedata.normalize('NFKC',text).splitlines():
        line = ' '.join(''.join(c for c in line if not unicodedata.category(c).startswith('C')).split())
        if line and any(c.isalnum() for c in line) and (not lines or line!=lines[-1]): lines.append(line)
    return '\n'.join(lines)

def parse(path):
    doc = pysubs2.load(str(path),encoding='utf-8')
    segments = []
    for index,event in enumerate(doc.events):
        if event.is_comment or event.is_drawing: continue
        text = event.plaintext.strip()
        if not text: continue
        if event.start < 0 or event.end <= event.start: raise ValueError('Invalid source subtitle timing')
        segments.append(SubtitleSegment(len(segments)+1,event.start/1000,event.end/1000,text,event_index=index))
    if not segments: raise ValueError('No usable subtitle segments found')
    return segments, doc

def extract(video, track, folder, config):
    suffix = 'ass' if track['codec_name'] in ('ass','ssa') else 'srt'
    path = folder / f'source.{suffix}'
    run([config.ffmpeg,'-nostdin','-y','-v','error','-i',video,'-map',f"0:{track['index']}",path],config.command_timeout)
    return parse(path)
