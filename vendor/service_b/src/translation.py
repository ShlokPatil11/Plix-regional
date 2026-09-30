import os
import re
from urllib.parse import urlsplit
from .common import PipelineError, request_json


def language(code):
    code = code.strip().lower().replace("_", "-")
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", code):
        raise PipelineError("Use language codes such as en, ml, hi or es-MX.")
    return code


class LibreTranslator:
    """Replaceable adapter. This provider translates individual cues, without a context API."""
    def __init__(self):
        self.url = os.environ.get("LIBRETRANSLATE_URL", "").rstrip("/")
        parts = urlsplit(self.url)
        if not parts.hostname or parts.username or parts.password:
            raise PipelineError("Set LIBRETRANSLATE_URL to your translation server URL.")
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise PipelineError("Use HTTPS for translation, or HTTP on localhost.")
        self.key = os.environ.get("LIBRETRANSLATE_API_KEY", "")

    def post(self, route, payload):
        if self.key:
            payload = {**payload, "api_key": self.key}
        return request_json(self.url + route, payload)

    def detect(self, cues):
        # Check every cue; uncertain or mixed input must not be silently misclassified.
        found = set()
        for cue in cues:
            candidates = self.post("/detect", {"q": cue.text})
            if not candidates:
                raise PipelineError("Source language detection failed; specify --source.")
            best = max(candidates, key=lambda item: item.get("confidence", 0))
            if best.get("confidence", 0) < 80:
                raise PipelineError(f"Uncertain source language at cue {cue.id}; review and specify --source.")
            found.add(best["language"])
        if len(found) != 1:
            raise PipelineError("Mixed-language input detected; review source text before translation.")
        return found.pop()

    def validate(self, source, target):
        supported = request_json(self.url + "/languages")
        entry = next((item for item in supported if item["code"] == source), None)
        if not entry or target not in entry.get("targets", []):
            raise PipelineError(f"Translation server does not support {source} -> {target}. Supply a reviewed translation JSON or another provider.")

    def translate(self, text, source, target, context=None, locale=None, glossary=None):
        if locale or glossary:
            raise PipelineError("This LibreTranslate adapter cannot enforce locale or glossary. Supply reviewed translations instead.")
        result = self.post("/translate", {"q": text, "source": source, "target": target, "format": "text"})
        translated = result.get("translatedText")
        if not isinstance(translated, str) or not translated.strip():
            raise PipelineError("Translation provider returned empty or invalid text.")
        return translated.strip()
