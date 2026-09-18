"""Internal HTTP microservice exposing ASR and Speech-to-Sign translation capabilities."""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# Ensure ml subpackages are on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "asr" / "src"))
sys.path.insert(0, str(ROOT / "translation" / "src"))

from asr.engine import StreamingAsrEngine
from translation.pipeline import SpeechToSignPipeline


class ModelServiceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for internal model inference RPCs."""

    pipeline = SpeechToSignPipeline()
    asr_engine = StreamingAsrEngine()

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path in ("/health", "/"):
            self._send_json(200, {
                "status": "ok",
                "service": "converse-ml-speech-to-sign",
                "version": "0.1.0",
            })
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError as e:
            self._send_json(400, {"ok": False, "error": f"Invalid JSON payload: {e}"})
            return

        if self.path == "/internal/speech-to-sign/compile":
            self._handle_compile(body)
        elif self.path == "/internal/speech/asr":
            self._handle_asr(body)
        elif self.path == "/internal/pipeline/audio-to-sign":
            self._handle_audio_to_sign(body)
        else:
            self._send_json(404, {"ok": False, "error": f"Endpoint not found: {self.path}"})

    def _handle_compile(self, body: dict[str, Any]) -> None:
        english_text = body.get("englishText", "")
        session_id = body.get("sessionId", "default_session")
        utterance_id = body.get("utteranceId", "utt_001")

        representation = self.pipeline.translate(
            english_text=english_text,
            session_id=session_id,
            utterance_id=utterance_id,
        )

        self._send_json(200, {
            "ok": True,
            "data": representation.to_dict(),
        })

    def _handle_asr(self, body: dict[str, Any]) -> None:
        audio_b64 = body.get("audioBase64", "")
        audio_format = body.get("audioFormat", "pcm_s16le")
        session_id = body.get("sessionId", "default_session")

        events = self.asr_engine.process_audio_chunk(
            session_id=session_id,
            audio_data=audio_b64,
            audio_format=audio_format,
        )

        flushed = self.asr_engine.flush_session(session_id)
        all_events = events + flushed

        final_transcript = ""
        confidence = 0.0
        duration_ms = 0.0

        if all_events:
            last = all_events[-1]
            final_transcript = last.text
            confidence = last.confidence
            duration_ms = last.latency_metrics.audio_duration_ms

        self._send_json(200, {
            "ok": True,
            "data": {
                "transcript": final_transcript,
                "isFinal": True,
                "confidence": confidence,
                "durationMs": duration_ms,
                "events": [e.to_dict() for e in all_events],
            },
        })

    def _handle_audio_to_sign(self, body: dict[str, Any]) -> None:
        audio_b64 = body.get("audioBase64", "")
        audio_format = body.get("audioFormat", "pcm_s16le")
        session_id = body.get("sessionId", "default_session")
        utterance_id = body.get("utteranceId", "utt_001")

        events = self.asr_engine.process_audio_chunk(
            session_id=session_id,
            audio_data=audio_b64,
            audio_format=audio_format,
        )
        flushed = self.asr_engine.flush_session(session_id)
        all_events = events + flushed

        transcript = all_events[-1].text if all_events else ""

        representation = self.pipeline.translate(
            english_text=transcript,
            session_id=session_id,
            utterance_id=utterance_id,
        )

        self._send_json(200, {
            "ok": True,
            "data": {
                "transcript": transcript,
                "representation": representation.to_dict(),
            },
        })


def run_server(port: int = 5050) -> None:
    server = HTTPServer(("127.0.0.1", port), ModelServiceHandler)
    print(f"Converse ML service running on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    run_server(port)
