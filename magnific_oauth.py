from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import requests


AUTH_ROOT = "https://auth.magnific.com/realms/mcp"
REGISTER_URL = f"{AUTH_ROOT}/clients-registrations/openid-connect"
AUTHORIZE_URL = f"{AUTH_ROOT}/protocol/openid-connect/auth"
TOKEN_URL = f"{AUTH_ROOT}/protocol/openid-connect/token"
MCP_URL = "https://mcp.magnific.com"
SCOPES = "openid profile email mcp:custom-audience offline_access"


class MagnificAuth:
    def __init__(self, storage_path: Path):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pending: dict[str, dict[str, str]] = {}
        self.lock = threading.RLock()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self, value: dict) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    def status(self) -> dict:
        data = self._read()
        tokens = data.get("tokens") or {}
        return {
            "connected": bool(
                tokens.get("refresh_token") or tokens.get("access_token")
            )
        }

    def begin(self, redirect_uri: str) -> str:
        with self.lock:
            saved = self._read()
            registration = saved.get("registration")
            if not registration or registration.get("redirect_uri") != redirect_uri:
                response = requests.post(REGISTER_URL, json={
                    "client_name": "Plix Directory Language Studio",
                    "redirect_uris": [redirect_uri],
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "token_endpoint_auth_method": "none",
                }, timeout=30)
                response.raise_for_status()
                registered = response.json()
                registration = {"client_id": registered["client_id"], "redirect_uri": redirect_uri}
                saved["registration"] = registration
                self._write(saved)
            verifier = secrets.token_urlsafe(64)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            state = secrets.token_urlsafe(24)
            self.pending[state] = {"verifier": verifier, **registration, "created": str(time.time())}
            return AUTHORIZE_URL + "?" + urlencode({
                "response_type": "code", "client_id": registration["client_id"],
                "redirect_uri": redirect_uri, "scope": SCOPES, "state": state,
                "code_challenge": challenge, "code_challenge_method": "S256", "resource": MCP_URL,
            })

    def finish(self, state: str, code: str) -> None:
        with self.lock:
            pending = self.pending.pop(state, None)
            if not pending:
                raise ValueError("Magnific login session expired; start the connection again")
            response = requests.post(TOKEN_URL, data={
                "grant_type": "authorization_code", "client_id": pending["client_id"],
                "redirect_uri": pending["redirect_uri"], "code": code,
                "code_verifier": pending["verifier"], "resource": MCP_URL,
            }, timeout=30)
            response.raise_for_status()
            saved = self._read()
            saved["tokens"] = response.json()
            self._write(saved)

    def access_token(self) -> str:
        with self.lock:
            saved = self._read()
            tokens = saved.get("tokens") or {}
            if tokens.get("refresh_token"):
                registration = saved.get("registration") or {}
                response = requests.post(TOKEN_URL, data={
                    "grant_type": "refresh_token", "client_id": registration.get("client_id"),
                    "refresh_token": tokens["refresh_token"], "resource": MCP_URL,
                }, timeout=30)
                if response.ok:
                    updated = response.json()
                    if "refresh_token" not in updated:
                        updated["refresh_token"] = tokens["refresh_token"]
                    saved["tokens"] = updated
                    self._write(saved)
                    return updated["access_token"]
            if tokens.get("access_token"):
                return tokens["access_token"]
            raise RuntimeError("Magnific is not connected. Connect it from the web UI first.")


class MagnificMcp:
    def __init__(self, token: str):
        self.http = requests.Session()
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"}
        self.session_id: str | None = None
        self.rpc_id = 0
        initialized = self._rpc("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "plix-directory-language-studio", "version": "1.0.0"},
        })
        if not initialized:
            raise RuntimeError("Magnific MCP initialization failed")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    @staticmethod
    def _decode(response: requests.Response) -> dict:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:].strip())
            return {}
        return response.json() if response.content else {}

    def _post(self, payload: dict) -> dict:
        headers = dict(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        response = self.http.post(MCP_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        self.session_id = response.headers.get("Mcp-Session-Id", self.session_id)
        return self._decode(response)

    def _rpc(self, method: str, params: dict) -> dict:
        self.rpc_id += 1
        body = self._post({"jsonrpc": "2.0", "id": self.rpc_id, "method": method, "params": params})
        if body.get("error"):
            raise RuntimeError(body["error"].get("message", str(body["error"])))
        return body.get("result") or {}

    def call(self, name: str, arguments: dict) -> dict:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            text = " ".join(item.get("text", "") for item in result.get("content", []))
            raise RuntimeError(text or f"Magnific tool {name} failed")
        if isinstance(result.get("structuredContent"), dict):
            return result["structuredContent"]
        for item in result.get("content", []):
            if item.get("type") == "text":
                try:
                    return json.loads(item["text"])
                except ValueError:
                    continue
        return result
