import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class PipelineError(Exception):
    """An actionable, safe-to-display job error."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows previewers/scanners can briefly hold the destination open.
            if attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_env(path):
    # Existing process variables take priority; deliberately no shell evaluation.
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def request_json(url, payload=None, headers=None, attempts=3):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", **(headers or {})})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # Do not echo provider bodies, which may contain credentials or source text.
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise PipelineError(f"Provider returned HTTP {error.code}; check account, settings and limits.") from None
            time.sleep(min(2 ** attempt, 8))
        except (urllib.error.URLError, TimeoutError):
            # POST timeout may already have incurred a charge; avoid blind paid retries.
            raise PipelineError("Provider connection failed. Resume after checking connectivity; a timed-out request may have been processed.") from None
        except (ValueError, UnicodeError):
            raise PipelineError("Provider returned invalid JSON.") from None
