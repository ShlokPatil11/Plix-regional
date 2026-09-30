from pathlib import Path
import cv2
import numpy as np
from .common import PipelineError


def detect_faces(image_bgr, min_size=(50, 50)):
    """
    Detects face-like regions using human skin color distribution in YCrCb,
    morphological filtering, and anatomical face aspect ratio constraints.
    Returns list of bounding boxes [(x, y, w, h), ...] sorted by size (largest first).
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    h_img, w_img = image_bgr.shape[:2]
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

    # Human skin threshold in YCrCb space
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Clean mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    min_w, min_h = min_size
    min_area = min_w * min_h

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        
        # Human face/head oval aspect ratio typically falls between 0.75 and 2.2
        if 0.75 <= aspect_ratio <= 2.2:
            # Check filling ratio (face oval fills roughly 45% - 90% of bounding box)
            fill_ratio = area / float(w * h)
            if 0.40 <= fill_ratio <= 0.95:
                candidates.append((x, y, w, h))

    if not candidates:
        return []

    # Sort descending by bounding box area (primary foreground face first)
    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    return candidates


def compute_skin_ratio(face_crop):
    """
    Calculates the proportion of pixels matching natural human skin tones
    in the YCrCb space.
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    skin_ratio = np.count_nonzero(mask) / (face_crop.shape[0] * face_crop.shape[1])
    return skin_ratio


def compute_face_signature(face_crop):
    """
    Computes an identity signature for character locking using normalized
    HSV color histogram and spatial intensity distribution.
    """
    if face_crop is None or face_crop.size == 0:
        return None
    resized = cv2.resize(face_crop, (100, 100))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # 2D Histogram for H and S channels (32 bins for Hue, 32 bins for Saturation)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def compare_face_signatures(sig1, sig2):
    """
    Compares two face signatures using correlation.
    Returns a score between -1.0 and 1.0 (typical match > 0.60).
    """
    if sig1 is None or sig2 is None:
        return 0.0
    return float(cv2.compareHist(sig1, sig2, cv2.HISTCMP_CORREL))


def extract_reference_signature_from_image(image_path):
    """
    Extracts the signature of the primary face from a reference image file.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise PipelineError(f"Could not read reference image: {image_path}")
    faces = detect_faces(img)
    if not faces:
        raise PipelineError("No face detected in reference image. Please provide a clear portrait.")
    x, y, w, h = faces[0]
    face_crop = img[y:y+h, x:x+w]
    return compute_face_signature(face_crop)


def classify_shot(video_path, start_sec, end_sec, reference_sig=None, samples=5, lock_threshold=0.55):
    """
    Analyzes a shot to determine if it is A-roll (target speaking character) or
    B-roll (mannequin, product demonstration, cutaway, bystander).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = end_sec - start_sec
    if duration <= 0:
        cap.release()
        return {"type": "b_roll", "reason": "Zero duration shot", "score": 0.0}

    timestamps = np.linspace(start_sec + 0.1 * duration, end_sec - 0.1 * duration, samples)
    face_detections = []
    skin_scores = []
    signatures = []

    for t in timestamps:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        faces = detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]
            crop = frame[y:y+h, x:x+w]
            face_detections.append(True)
            skin_scores.append(compute_skin_ratio(crop))
            signatures.append(compute_face_signature(crop))
        else:
            face_detections.append(False)

    cap.release()

    face_ratio = sum(face_detections) / max(1, len(face_detections))

    # Condition 1: No human face detected in most samples -> Definite B-roll
    if face_ratio < 0.4:
        return {
            "type": "b_roll",
            "reason": f"No face detected ({face_ratio*100:.0f}% frames with face) - B-roll cutaway/product",
            "score": 0.0,
            "face_ratio": face_ratio
        }

    # Condition 2: Skin tone check for mannequins / plastic figures
    avg_skin = float(np.mean(skin_scores)) if skin_scores else 0.0
    if avg_skin < 0.15:
        return {
            "type": "b_roll",
            "reason": f"Non-human skin texture/material detected (skin ratio {avg_skin:.2f}) - Mannequin or artwork",
            "score": avg_skin,
            "face_ratio": face_ratio
        }

    # Condition 3: Character lock check if reference signature provided
    if reference_sig is not None and signatures:
        match_scores = [compare_face_signatures(reference_sig, sig) for sig in signatures if sig is not None]
        avg_match = float(np.mean(match_scores)) if match_scores else 0.0
        if avg_match < lock_threshold:
            return {
                "type": "b_roll",
                "reason": f"Face does not match target character (similarity {avg_match:.2f} < {lock_threshold}) - Secondary person/bystander",
                "score": avg_match,
                "face_ratio": face_ratio
            }
        else:
            return {
                "type": "a_roll",
                "reason": f"Target character confirmed (similarity {avg_match:.2f})",
                "score": avg_match,
                "face_ratio": face_ratio
            }

    # Default to A-roll if face is consistently present and human
    return {
        "type": "a_roll",
        "reason": f"Human face detected ({face_ratio*100:.0f}% frames)",
        "score": avg_skin,
        "face_ratio": face_ratio
    }


def auto_detect_creator_anchor(video_path, scenes=None, sample_stride=4, min_motion=5.0):
    """
    Scans candidate video scenes to automatically extract the facial identity
    signature of the primary speaker (character anchor) without requiring
    a manual portrait photo.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not scenes:
        scenes = [{"start": 0.0, "end": total_frames / fps, "duration": total_frames / fps}]

    best_candidate = None
    best_score = -1.0

    for sc in scenes:
        s = sc.get("start", 0.0)
        e = sc.get("end", total_frames / fps)
        start_f = int(s * fps)
        end_f = int(e * fps)
        if end_f <= start_f:
            continue

        face_crops = []
        mouth_diffs = []
        skin_vals = []
        centered_cnt = 0
        prev_mouth = None

        for f_idx in range(start_f, end_f, sample_stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Fast downscale for high-res frames to speed up analysis
            h_orig, w_orig = frame.shape[:2]
            if w_orig > 720:
                scale = 720.0 / w_orig
                proc_frame = cv2.resize(frame, (720, int(h_orig * scale)))
            else:
                proc_frame = frame
                scale = 1.0

            faces = detect_faces(proc_frame)
            if faces:
                x_p, y_p, w_p, h_p = faces[0]
                # Map back to original dimensions for crop
                x = int(x_p / scale)
                y = int(y_p / scale)
                w = int(w_p / scale)
                h = int(h_p / scale)

                cx = (x + w / 2) / frame.shape[1]
                cy = (y + h / 2) / frame.shape[0]
                if 0.30 <= cx <= 0.70 and 0.30 <= cy <= 0.75:
                    centered_cnt += 1
                crop = frame[y:y+h, x:x+w]
                skin = compute_skin_ratio(crop)
                skin_vals.append(skin)
                face_crops.append(crop)

                mouth_y = int(y + 0.65 * h)
                mouth_crop = frame[mouth_y:y+h, x:x+w]
                if mouth_crop.size > 0:
                    mouth_gray = cv2.cvtColor(cv2.resize(mouth_crop, (64, 32)), cv2.COLOR_BGR2GRAY)
                    if prev_mouth is not None:
                        mouth_diffs.append(np.mean(cv2.absdiff(prev_mouth, mouth_gray)))
                    prev_mouth = mouth_gray

        total_samples = max(1, (end_f - start_f) // sample_stride)
        if not face_crops:
            continue

        face_ratio = len(face_crops) / total_samples
        center_ratio = centered_cnt / total_samples
        avg_skin = float(np.mean(skin_vals)) if skin_vals else 0.0
        avg_motion = float(np.mean(mouth_diffs)) if mouth_diffs else 0.0

        # Talking head speaker criteria: centered face, natural human skin, and moderate speech articulation (5-28)
        if face_ratio >= 0.40 and center_ratio >= 0.40 and (0.20 <= avg_skin <= 0.85) and (min_motion <= avg_motion <= 28.0):
            score = sc.get("duration", (e - s)) * min(face_ratio, 1.0) * min(center_ratio, 1.0)
            if score > best_score:
                best_score = score
                best_candidate = face_crops[len(face_crops) // 2]

    cap.release()
    if best_candidate is not None:
        return compute_face_signature(best_candidate)
    return None


def classify_shot_voiceover_aware(
    video_path,
    start_sec,
    end_sec,
    reference_sig=None,
    motion_threshold=7.5,
    match_threshold=0.45,
    sample_stride=4
):
    """
    Analyzes a video shot to differentiate between active A-roll talking heads (requiring lip-sync)
    and voiceover B-roll demonstrations (product close-ups, droppers, foam washing, hands covering face,
    CGI illustrations, or silent posing).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "type": "a_roll",
            "action": "sync",
            "reason": "Could not open video file for visual inspection",
            "face_ratio": 1.0,
            "match": 1.0,
            "motion": 10.0,
            "skin": 0.5
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_f = int(start_sec * fps)
    end_f = int(end_sec * fps)
    total_samples = max(1, (end_f - start_f) // sample_stride)

    if end_f <= start_f:
        cap.release()
        return {
            "type": "b_roll",
            "action": "passthrough",
            "reason": "Zero duration shot",
            "face_ratio": 0.0,
            "match": 0.0,
            "motion": 0.0,
            "skin": 0.0
        }

    face_counts = 0
    match_scores = []
    mouth_diffs = []
    skin_scores = []
    prev_mouth = None

    for f_idx in range(start_f, end_f, sample_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Fast downscale for high-res frames
        h_orig, w_orig = frame.shape[:2]
        if w_orig > 720:
            scale = 720.0 / w_orig
            proc_frame = cv2.resize(frame, (720, int(h_orig * scale)))
        else:
            proc_frame = frame
            scale = 1.0

        faces = detect_faces(proc_frame)
        if faces:
            face_counts += 1
            x_p, y_p, w_p, h_p = faces[0]
            x = int(x_p / scale)
            y = int(y_p / scale)
            w = int(w_p / scale)
            h = int(h_p / scale)

            crop = frame[y:y+h, x:x+w]
            skin_scores.append(compute_skin_ratio(crop))

            if reference_sig is not None:
                sig = compute_face_signature(crop)
                if sig is not None:
                    match_scores.append(compare_face_signatures(reference_sig, sig))

            mouth_y = int(y + 0.65 * h)
            mouth_crop = frame[mouth_y:y+h, x:x+w]
            if mouth_crop.size > 0:
                mouth_gray = cv2.cvtColor(cv2.resize(mouth_crop, (64, 32)), cv2.COLOR_BGR2GRAY)
                if prev_mouth is not None:
                    mouth_diffs.append(np.mean(cv2.absdiff(prev_mouth, mouth_gray)))
                prev_mouth = mouth_gray

    cap.release()

    face_ratio = face_counts / float(total_samples)
    avg_skin = float(np.mean(skin_scores)) if skin_scores else 0.0
    avg_match = float(np.mean(match_scores)) if match_scores else 0.0
    avg_motion = float(np.mean(mouth_diffs)) if mouth_diffs else 0.0

    # Decision rules:
    if face_ratio < 0.40:
        return {
            "type": "b_roll",
            "action": "passthrough",
            "reason": f"No face detected in majority of frames ({face_ratio*100:.0f}%) - B-roll product/cutaway",
            "face_ratio": face_ratio,
            "match": avg_match,
            "motion": avg_motion,
            "skin": avg_skin
        }

    if avg_skin < 0.15:
        return {
            "type": "b_roll",
            "action": "passthrough",
            "reason": f"Non-human skin texture ({avg_skin:.2f}) - Mannequin or artwork",
            "face_ratio": face_ratio,
            "match": avg_match,
            "motion": avg_motion,
            "skin": avg_skin
        }

    if reference_sig is not None and avg_match < match_threshold:
        return {
            "type": "b_roll",
            "action": "passthrough",
            "reason": f"Face does not match speaker ({avg_match:.2f} < {match_threshold:.2f}) - B-roll demonstration or bystander",
            "face_ratio": face_ratio,
            "match": avg_match,
            "motion": avg_motion,
            "skin": avg_skin
        }

    if avg_motion < motion_threshold:
        return {
            "type": "b_roll",
            "action": "passthrough",
            "reason": f"Mouth motion below speaking threshold ({avg_motion:.1f} < {motion_threshold:.1f}) - Silent voiceover posing",
            "face_ratio": face_ratio,
            "match": avg_match,
            "motion": avg_motion,
            "skin": avg_skin
        }

    if avg_motion > 30.0:
        return {
            "type": "b_roll",
            "action": "passthrough",
            "reason": f"Rapid cutaway / hand movement ({avg_motion:.1f} > 30.0) - B-roll demonstration",
            "face_ratio": face_ratio,
            "match": avg_match,
            "motion": avg_motion,
            "skin": avg_skin
        }

    return {
        "type": "a_roll",
        "action": "sync",
        "reason": f"Talking head confirmed (match: {avg_match:.2f}, motion: {avg_motion:.1f})",
        "face_ratio": face_ratio,
        "match": avg_match,
        "motion": avg_motion,
        "skin": avg_skin
    }

