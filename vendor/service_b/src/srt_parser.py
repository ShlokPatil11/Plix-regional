import html
import re
from dataclasses import asdict, dataclass
from .common import PipelineError

STAMP = r"(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})"
TIMING = re.compile(rf"^{STAMP}\s*-->\s*{STAMP}$")


@dataclass
class Cue:
    id: int
    start_ms: int
    end_ms: int
    text: str

    def to_dict(self):
        return asdict(self)


def parse_srt(text, preserve_lines=False):
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise PipelineError("SRT is empty.")
    cues, ids = [], set()
    for block in re.split(r"\n[ \t]*\n", text):
        lines = block.strip().splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise PipelineError("Each SRT cue needs an integer ID, timestamp line and text.")
        cue_id = int(lines[0].strip())
        match = TIMING.fullmatch(lines[1].strip())
        if not match:
            raise PipelineError(f"Invalid timestamps in cue {cue_id}. Use HH:MM:SS,mmm.")
        values = list(map(int, match.groups()))
        def ms(parts):
            h, m, s, fraction = parts
            return ((h * 60 + m) * 60 + s) * 1000 + fraction
        start, end = ms(values[:4]), ms(values[4:])
        cleaned_lines = [" ".join(html.unescape(re.sub(r"<[^>]+>", "", line)).split()) for line in lines[2:]]
        clean = "\n".join(cleaned_lines) if preserve_lines else " ".join(" ".join(cleaned_lines).split())
        if cue_id in ids or cue_id < 1 or end <= start or not clean.strip():
            raise PipelineError(f"Duplicate/invalid ID, empty text, or non-positive window in cue {cue_id}.")
        if cues and start < cues[-1].end_ms:
            raise PipelineError(f"Cue {cue_id} overlaps or precedes the previous cue; resolve the single-narrator timeline.")
        ids.add(cue_id)
        cues.append(Cue(cue_id, start, end, clean))
    return cues


def stamp(ms):
    h, remainder = divmod(int(ms), 3600000)
    m, remainder = divmod(remainder, 60000)
    s, fraction = divmod(remainder, 1000)
    return f"{h:02}:{m:02}:{s:02},{fraction:03}"


def render_srt(segments):
    return "\n\n".join(
        f"{index}\n{stamp(s['start_ms'])} --> {stamp(s['end_ms'])}\n{s['target_text']}"
        for index, s in enumerate(segments, 1) if not s.get("skip")
    ) + "\n"
