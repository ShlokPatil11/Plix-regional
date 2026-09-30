TEXT_CODECS = {'ass','ssa','subrip','mov_text','webvtt','text','srt'}

def detect(metadata, config):
    tracks = metadata.subtitle_streams
    if not tracks: return None
    if config.subtitle_stream is not None:
        selected = next((s for s in tracks if s['index']==config.subtitle_stream),None)
        if selected is None: raise ValueError('SUBTITLE_STREAM index is not present')
    else:
        selected = next((s for s in tracks if s.get('tags',{}).get('language')==config.source_language),None)
        selected = selected or next((s for s in tracks if s.get('disposition',{}).get('default')),tracks[0])
    if selected['codec_name'] not in TEXT_CODECS:
        raise ValueError('Selected embedded subtitles are bitmap-based. Convert the track to text with OCR, or select a text SUBTITLE_STREAM.')
    return selected
