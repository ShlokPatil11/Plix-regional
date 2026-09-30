import json
import math
from pipeline.video_analyzer import analyze
from pipeline.translator import visual_length

def check(source, output, originals, translated, config, report_path, warnings=None):
    errors=[]; warnings=list(warnings or [])
    report={'status':'FAIL','source_duration':source.duration,'output_duration':None,
            'subtitle_count':len(originals),'translated_count':len(translated),'warnings':warnings,'errors':errors}
    try:
        if not output.is_file() or output.stat().st_size==0: raise ValueError('Output missing or empty')
        actual=analyze(output,config)
        report['output_duration']=actual.duration
        if abs(source.duration-actual.duration)>max(0.25,2/max(source.fps,1)): errors.append('Duration mismatch')
        if (source.width,source.height)!=(actual.width,actual.height): errors.append('Dimensions changed')
        if source.fps and abs(source.fps-actual.fps)>0.1: errors.append('Average FPS changed')
        if source.sample_aspect_ratio!=actual.sample_aspect_ratio: errors.append('Sample aspect ratio changed')
        if len(source.audio_streams)!=len(actual.audio_streams): errors.append('Audio stream count changed')
        for old,new in zip(source.audio_streams,actual.audio_streams):
            if any(old.get(k)!=new.get(k) for k in ('codec_name','sample_rate','channels')): errors.append('Audio properties changed')
        if [s.id for s in originals]!=[s.id for s in translated]: errors.append('Subtitle ID/count mismatch')
        for original,segment in zip(originals,translated):
            if (original.start,original.end)!=(segment.start,segment.end): errors.append(f'Timing changed: {segment.id}')
            if not all(math.isfinite(t) for t in (segment.start,segment.end)) or not 0<=segment.start<segment.end<=source.duration+0.1:
                errors.append(f'Invalid timing: {segment.id}')
            text=segment.translated_text or ''
            if not text.strip(): errors.append(f'Empty translation: {segment.id}')
            if len(text.splitlines())>config.max_lines or any(visual_length(line)>config.max_chars for line in text.splitlines()): errors.append(f'Text exceeds limits: {segment.id}')
            box=segment.bbox
            if box and (min(box.x,box.y)<0 or min(box.width,box.height)<=0 or box.x+box.width>source.width or box.y+box.height>source.height): errors.append(f'Bounding box outside frame: {segment.id}')
        # Exact timing equality above also guarantees preservation of overlap relationships.
        warnings.append('Bounds checks validate detected boxes; exact rendered glyph extents and complex ASS animations require visual review.')
    except Exception as exc:
        errors.append(str(exc))
    report['status']='PASS' if not errors else 'FAIL'
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    if errors: raise ValueError('QC failed: '+'; '.join(errors))
    return report
