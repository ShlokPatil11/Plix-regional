import json
import logging
import re
import subprocess
import unicodedata
from pathlib import Path

class MediaError(RuntimeError): pass

def run(args, timeout=120, cwd=None):
    logging.getLogger(__name__).info('Command arguments: %r', [str(a) for a in args])
    try:
        result = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=timeout, shell=False)
    except FileNotFoundError as exc:
        raise MediaError(f'Executable missing: {args[0]}. Install FFmpeg and set FFMPEG_PATH/FFPROBE_PATH.') from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f'Command timed out after {timeout}s: {args[0]}') from exc
    if result.returncode:
        raise MediaError(f'{args[0]} exited {result.returncode}\n{result.stderr}')
    return result.stdout

def probe(path, config, timeout=30):
    return json.loads(run([config.ffprobe,'-v','error','-show_format','-show_streams','-of','json',path],timeout))

def output_filename(filename, language):
    def clean(value):
        cleaned=''.join(c if unicodedata.category(c)[0] in 'LMN' or c in ' .-_' else '_' for c in value)
        cleaned=cleaned.strip(' .')[:100] or 'video'
        if cleaned.upper().split('.')[0] in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}: cleaned='_'+cleaned
        return cleaned
    return f'{clean(Path(filename).stem)}_{clean(language)}.mp4'
