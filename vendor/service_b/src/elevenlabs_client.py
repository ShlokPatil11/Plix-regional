import base64
import os
from urllib.parse import quote
from .common import PipelineError, request_json, write_json, read_json, digest


class ElevenLabs:
    def __init__(self, model, voice, target):
        key = os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise PipelineError("Set ELEVENLABS_API_KEY in .env or your environment.")
        if not voice:
            raise PipelineError("Specify an ElevenLabs voice ID.")
        self.headers = {"xi-api-key": key}
        self.model, self.voice, self.target = model, voice, target
        self.limit = None

    def preflight(self):
        models = request_json("https://api.elevenlabs.io/v1/models", headers=self.headers)
        model = next((m for m in models if m["model_id"] == self.model), None)
        if not model or not model.get("can_do_text_to_speech"):
            raise PipelineError("Selected ElevenLabs model does not support text to speech.")
        codes = {x["language_id"].lower() for x in model.get("languages", [])}
        if self.target not in codes:
            raise PipelineError(f"Model {self.model} does not advertise language {self.target}. Select a supported model/language code.")
        self.limit = model.get("maximum_text_length_per_request") or model.get("max_characters_request_free_user")
        request_json("https://api.elevenlabs.io/v1/voices/" + quote(self.voice, safe=""), headers=self.headers)

    def generate(self, text, cache, revision=0):
        if self.limit and len(text) > self.limit:
            raise PipelineError(f"Text exceeds model request limit ({self.limit} characters); split this cue.")
        payload = {"text": text, "model_id": self.model, "language_code": self.target}
        key = digest({"provider": "elevenlabs", "voice": self.voice, "payload": payload,
                      "format": "mp3_44100_128", "revision": revision})
        path = cache / (key + ".json")
        hit = path.exists()
        response = read_json(path) if hit else request_json(
            "https://api.elevenlabs.io/v1/text-to-speech/" + quote(self.voice, safe="") +
            "/with-timestamps?output_format=mp3_44100_128", payload, self.headers)
        try:
            audio = base64.b64decode(response["audio_base64"], validate=True)
            if not audio:
                raise ValueError()
        except (KeyError, ValueError, TypeError):
            raise PipelineError("ElevenLabs returned invalid audio.") from None
        if not hit:
            write_json(path, response)
        return audio, response.get("normalized_alignment") or response.get("alignment"), hit
