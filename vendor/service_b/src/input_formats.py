"""Import supplied translations without contacting any translation provider."""
from pathlib import Path
from .common import PipelineError
from .srt_parser import Cue, parse_srt


def split_bilingual(raw, source_lines=1, source_first=True):
    """Split by an explicit layout; never infer language from alphabet/script."""
    if type(source_lines) is not int or source_lines < 1:
        raise PipelineError("Source-line count must be a positive integer.")
    cues = parse_srt(raw, preserve_lines=True)
    sources, targets = [], {}
    for cue in cues:
        lines = cue.text.splitlines()
        if len(lines) <= source_lines:
            raise PipelineError(f"Cue {cue.id} needs source and translated lines. Check the layout or use separate SRT files.")
        if source_first:
            original, translated = lines[:source_lines], lines[source_lines:]
        else:
            original, translated = lines[-source_lines:], lines[:-source_lines]
        original, translated = " ".join(original).strip(), " ".join(translated).strip()
        if not original or not translated:
            raise PipelineError(f"Cue {cue.id} has an empty source or translation.")
        sources.append(Cue(cue.id, cue.start_ms, cue.end_ms, original))
        targets[str(cue.id)] = translated
    return sources, targets


def pair_translated_srt(cues, path):
    raw = Path(path).read_text(encoding="utf-8-sig")
    translated = parse_srt(raw)
    if len(cues) != len(translated):
        raise PipelineError("Source and translated SRT files must have the same number of cues.")
    result = {}
    for source, target in zip(cues, translated):
        if (source.start_ms, source.end_ms) != (target.start_ms, target.end_ms):
            raise PipelineError(f"Translated SRT timestamps do not match source cue {source.id}. Align the files before importing.")
        # Cue numbering may differ between files; pair by validated timeline position.
        result[str(source.id)] = target.text
    return result, raw


def split_labeled_bilingual(raw, source_label, target_label):
    """Select text by explicit [Language]: markers, independently of line order."""
    import re
    source_key, target_key = source_label.strip().casefold(), target_label.strip().casefold()
    if not source_key or not target_key or source_key == target_key:
        raise PipelineError("Source and target labels must be distinct nonempty language labels.")
    marker = re.compile(r"^\[([^\]]+)\]\s*:\s*(.*)$")
    cues = parse_srt(raw, preserve_lines=True)
    sources, targets = [], {}
    for cue in cues:
        sections = {}
        current = None
        for line in cue.text.splitlines():
            match = marker.match(line.strip())
            if match:
                current = match[1].strip().casefold()
                if current not in (source_key, target_key):
                    raise PipelineError(f"Unexpected language label [{match[1]}] in cue {cue.id}.")
                if current in sections:
                    raise PipelineError(f"Duplicate language label in cue {cue.id}.")
                sections[current] = [match[2].strip()]
            elif current is not None:
                sections[current].append(line.strip())
            elif line.strip():
                raise PipelineError(f"Unlabeled text before the first language label in cue {cue.id}.")
        if set(sections) != {source_key, target_key}:
            raise PipelineError(f"Cue {cue.id} must contain [{source_label}]: and [{target_label}]:.")
        source = " ".join(" ".join(sections[source_key]).split())
        target = " ".join(" ".join(sections[target_key]).split())
        if not source or not target:
            raise PipelineError(f"Cue {cue.id} has empty source or translated text.")
        sources.append(Cue(cue.id, cue.start_ms, cue.end_ms, source))
        targets[str(cue.id)] = target
    return sources, targets
