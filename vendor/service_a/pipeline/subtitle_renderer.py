import shutil
from models.subtitle import BoundingBox
from pipeline.ffmpeg_utils import run
from pipeline.burned_subtitle_ocr import OverlayRemoval, InpaintingRemoval, DelogoRemoval

def render_pill_overlays(video, segments, config, folder):
    from PIL import Image, ImageDraw, ImageFont
    from pipeline.subtitle_style import choose_font, resolve_font_file
    
    font_file = None
    target_lang = getattr(config, 'target_language', '').lower()
    candidates = [*config.font_dir.glob('*.ttf'), *config.font_dir.glob('*.otf')]
    preferred_file = resolve_font_file(config, target_lang)
    if preferred_file:
        font_file = str(preferred_file)
    if not font_file and target_lang:
        for c in candidates:
            if target_lang in c.stem.lower():
                font_file = str(c)
                break
    if not font_file:
        font_name = choose_font(config)
        for c in candidates:
            if c.stem.casefold() == font_name.casefold() or font_name.casefold() in c.stem.casefold():
                font_file = str(c)
                break
    if not font_file and candidates:
        font_file = str(candidates[0])

    inputs = ['-i', str(video)]
    filter_parts = []
    last_v = '0:v'
    card_idx = 0

    for seg in segments:
        if not seg.bbox or not seg.translated_text:
            continue
        card_idx += 1
        box = seg.bbox
        bw, bh = box.width, box.height
        bx, by = box.x, box.y

        patch_hex = box.patch_color or getattr(config, 'overlay_patch_color', '0x000000')
        clean = patch_hex.lstrip('#').replace('0x', '')
        if len(clean) == 6:
            r, g, b = int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)
        else:
            r, g, b = 0, 0, 0
        fill_color = (r, g, b, 255)

        txt_hex = getattr(config, 'box_text_color', '#FFFFFF')
        clean_t = txt_hex.lstrip('#').replace('0x', '')
        if len(clean_t) == 6:
            tr, tg, tb = int(clean_t[0:2], 16), int(clean_t[2:4], 16), int(clean_t[4:6], 16)
        else:
            tr, tg, tb = 255, 255, 255
        text_color = (tr, tg, tb, 255)

        img = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # An opaque rectangle covers every source pixel in the detected box,
        # including the corners where a rounded transparent overlay could leak text.
        d.rectangle([0, 0, bw, bh], fill=fill_color)

        # Auto-fit font size to pill dimensions
        f_size = int(getattr(seg.style, 'font_size', 0) or getattr(config, 'font_size', 0) or 46)
        text = (seg.translated_text or '').strip()
        lines = text.splitlines() or ['']
        
        max_allowed_w = bw * 0.92
        max_allowed_h = bh * 0.82
        
        font = ImageFont.truetype(font_file, f_size) if font_file else ImageFont.load_default(f_size)
        while f_size > 16:
            max_w = max((d.textbbox((0, 0), l, font=font)[2] - d.textbbox((0, 0), l, font=font)[0]) for l in lines)
            spacing = 6 if len(lines) > 1 else 0
            tot_h = sum((d.textbbox((0, 0), l, font=font)[3] - d.textbbox((0, 0), l, font=font)[1]) for l in lines) + (len(lines) - 1) * spacing
            if max_w <= max_allowed_w and tot_h <= max_allowed_h:
                break
            f_size -= 2
            font = ImageFont.truetype(font_file, f_size) if font_file else ImageFont.load_default(f_size)
        spacing = 6 if len(lines) > 1 else 0
        total_h = sum((d.textbbox((0, 0), l, font=font)[3] - d.textbbox((0, 0), l, font=font)[1]) for l in lines) + (len(lines) - 1) * spacing
        cur_y = (bh - total_h) / 2 - 2
        for l in lines:
            tb = d.textbbox((0, 0), l, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            tx = (bw - tw) / 2 - tb[0]
            d.text((tx, cur_y - tb[1]), l, font=font, fill=text_color)
            cur_y += th + spacing

        card_path = folder / f'card_{card_idx}.png'
        img.save(str(card_path))

        inputs.extend(['-i', str(card_path)])
        out_v = f'v{card_idx}'
        filter_parts.append(f"[{last_v}][{card_idx}:v]overlay=x={bx}:y={by}:enable='gte(t,{seg.start:.3f})*lt(t,{seg.end:.3f})'[{out_v}]")
        last_v = out_v

    if card_idx == 0:
        raise RuntimeError('No segments with valid boxes and translations to render with pill overlay')

    filter_str = ';'.join(filter_parts)
    output = folder / 'rendered.mp4'
    cmd = [
        config.ffmpeg, '-nostdin', '-y', '-v', 'warning',
        *inputs,
        '-filter_complex', filter_str,
        '-map', f'[{last_v}]', '-map', '0:a?',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-fps_mode', 'passthrough',
        '-c:a', 'copy', '-map_metadata', '0', '-movflags', '+faststart',
        str(output)
    ]
    run(cmd, config.command_timeout, cwd=folder)
    return output

def render(video, subtitles, segments, config, folder, burned=False, metadata=None):
    use_pill = burned and getattr(config, 'pill_mode', False) and any(s.bbox for s in segments) and any(s.translated_text for s in segments)
    if use_pill:
        try:
            return render_pill_overlays(video, segments, config, folder)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning('Pill overlay rendering failed, falling back to drawbox: %s', exc)

    filters=[]
    if burned:
        if config.removal_mode not in ('overlay','inpainting', 'delogo'): raise ValueError('REMOVAL_MODE must be overlay, delogo, or inpainting')
        if config.removal_mode == 'delogo':
            removal = DelogoRemoval()
        elif config.removal_mode == 'inpainting':
            removal = InpaintingRemoval()
        else:
            patch_color = getattr(config, 'overlay_patch_color', '0xF4D9B9')
            removal = OverlayRemoval(patch_color)
        patch_spans = []
        for segment in segments:
            if not config.overlay_patch and config.removal_mode not in ('inpainting', 'delogo'):
                continue
            box = segment.bbox
            if not box:
                continue
            if patch_spans:
                last_box, last_start, last_end = patch_spans[-1]
                if (last_box.x == box.x and last_box.y == box.y and
                    last_box.width == box.width and last_box.height == box.height and
                    segment.start <= last_end + 0.15):
                    patch_spans[-1] = (last_box, last_start, max(last_end, segment.end))
                    continue
            patch_spans.append((box, segment.start, segment.end))

        # Back-fill: if top-area segments exist before the first bottom-area mask,
        # add a full-width bottom mask for that uncovered gap so original text is hidden
        if metadata and patch_spans:
            frame_h = getattr(metadata, 'height', 0)
            frame_w = getattr(metadata, 'width', 0)
            if frame_h and frame_w:
                bottom_spans = [(box, s, e) for box, s, e in patch_spans if box.y > frame_h * 0.4]
                top_spans = [(box, s, e) for box, s, e in patch_spans if box.y <= frame_h * 0.4]
                if bottom_spans and top_spans:
                    first_bottom_start = min(s for _, s, _ in bottom_spans)
                    top_covers_before_bottom = any(s < first_bottom_start for _, s, _ in top_spans)
                    if top_covers_before_bottom and first_bottom_start > 0.5:
                        first_bottom_box = min(bottom_spans, key=lambda t: t[1])[0]
                        extra_box = BoundingBox(0, first_bottom_box.y, frame_w, first_bottom_box.height,
                                               patch_color=first_bottom_box.patch_color)
                        patch_spans.insert(0, (extra_box, 0.0, first_bottom_start))

        for box, start, end in patch_spans:
            filters.append(removal.remove_original_subtitle(box, (start, end)))
    # Fixed relative names avoid FFmpeg filtergraph escaping of user-controlled paths.
    fonts=folder/'render_fonts'
    fonts.mkdir(exist_ok=True)
    for font in config.font_dir.iterdir():
        if font.suffix.lower() in ('.ttf','.otf') and font.is_file(): shutil.copy2(font,fonts/font.name)
    filters.append(f"subtitles=filename={subtitles.name}:fontsdir=render_fonts")
    script=folder/'render.ffscript'
    script.write_text(','.join(filters),encoding='utf-8')
    output=folder/'rendered.mp4'
    run([config.ffmpeg,'-nostdin','-y','-v','warning','-i',video,
         '-map','0:V:0','-map','0:a?','-/filter:v',script.name,
         '-c:v','libx264','-crf','18','-preset','medium','-fps_mode','passthrough',
         '-c:a','copy','-map_metadata','0','-movflags','+faststart',output],config.command_timeout,cwd=folder)
    try:
        shutil.rmtree(fonts)
    except Exception:
        pass
    return output
