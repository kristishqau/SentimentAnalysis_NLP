"""Small dependency-free web demo for a trusted local model artifact."""

# ruff: noqa: E501 -- embedded HTML/CSS/JavaScript is kept readable as a single asset.

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_analysis.artifacts import load_artifact

MAX_BODY_BYTES = 16_384
MAX_TEXT_LENGTH = 2_000

DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Entity Sentiment Lab</title>
  <style>
    :root { color-scheme: dark; --ink:#f7f7f2; --muted:#a8b3aa; --lime:#c9ff3d;
      --panel:#151a18; --line:#2c3530; --negative:#ff6b68; --positive:#6ee7a8;
      --neutral:#8dc8ff; --irrelevant:#d2a8ff; }
    * { box-sizing:border-box } body { margin:0; min-height:100vh; color:var(--ink);
      background:radial-gradient(circle at 80% 0,#263326 0,transparent 34%),#0b0e0c;
      font:16px/1.5 Inter,ui-sans-serif,system-ui,sans-serif; }
    main { width:min(1040px,calc(100% - 32px)); margin:auto; padding:64px 0; }
    header { display:grid; grid-template-columns:1.5fr 1fr; gap:32px; align-items:end;
      margin-bottom:36px } .eyebrow { color:var(--lime); text-transform:uppercase;
      letter-spacing:.16em; font-size:.75rem; font-weight:800 }
    h1 { font-size:clamp(2.8rem,8vw,6.2rem); line-height:.88; letter-spacing:-.07em;
      margin:.18em 0 } header p { color:var(--muted); max-width:36rem }
    .grid { display:grid; grid-template-columns:1.25fr .75fr; gap:20px }
    .card { background:color-mix(in srgb,var(--panel) 94%,transparent); border:1px solid var(--line);
      border-radius:20px; padding:24px; box-shadow:0 24px 70px #0006 }
    label { display:block; color:var(--muted); font-size:.78rem; font-weight:700;
      text-transform:uppercase; letter-spacing:.12em; margin:0 0 8px }
    input,textarea { width:100%; border:1px solid var(--line); border-radius:12px;
      background:#0d110f; color:var(--ink); padding:13px 14px; font:inherit; outline:none }
    input:focus,textarea:focus { border-color:var(--lime); box-shadow:0 0 0 3px #c9ff3d22 }
    textarea { min-height:170px; resize:vertical } .field { margin-bottom:18px }
    button { border:0; border-radius:999px; background:var(--lime); color:#10140d;
      font-weight:850; padding:13px 20px; cursor:pointer } button:disabled { opacity:.55 }
    .examples { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px }
    .examples button { background:#222a25; color:var(--ink); padding:8px 12px; font-size:.82rem }
    #result { min-height:96px; display:grid; align-content:center }
    .label { font-size:2rem; font-weight:900; letter-spacing:-.04em }
    .confidence { color:var(--muted); margin-top:6px }
    .meter { height:7px; background:#262d29; border-radius:10px; overflow:hidden; margin-top:14px }
    .meter span { display:block; height:100%; background:var(--lime); width:0; transition:width .4s }
    h2 { margin:0 0 12px; font-size:1rem } ul { padding-left:18px; color:var(--muted) }
    .status { display:inline-flex; align-items:center; gap:8px; color:var(--muted); font-size:.8rem }
    .status::before { content:""; width:8px; height:8px; border-radius:50%; background:var(--positive) }
    footer { margin-top:20px; color:#778078; font-size:.78rem }
    @media(max-width:760px){header,.grid{grid-template-columns:1fr}main{padding:36px 0}}
  </style>
</head>
<body><main>
  <header><div><div class="eyebrow">Leakage-safe NLP</div><h1>Entity<br>Sentiment Lab</h1></div>
    <p>Explore a four-class classifier trained with connected-group evaluation.
    Entity context matters: the same sentence can express different sentiment toward different targets.</p></header>
  <div class="grid"><section class="card">
    <div class="field"><label for="entity">Target entity</label><input id="entity" value="Microsoft" maxlength="120"></div>
    <div class="field"><label for="text">Social post</label><textarea id="text" maxlength="2000">The latest update is excellent and everything works smoothly.</textarea></div>
    <button id="analyze">Analyze sentiment</button>
    <div class="examples"><button data-example="positive">Positive example</button><button data-example="negative">Negative example</button><button data-example="relevance">Entity relevance</button></div>
  </section><aside>
    <section class="card" id="result"><div class="status">Model ready</div><div>Select Analyze to run a local prediction.</div></section>
    <section class="card" style="margin-top:20px"><h2>Use responsibly</h2><ul>
      <li>Confidence is not calibrated probability.</li><li>Short text and Irrelevant are weakest.</li>
      <li>Not for moderation or decisions about people.</li><li>Input stays on this local server.</li>
    </ul></section>
  </aside></div>
  <footer>TF-IDF + entity context · balanced logistic regression · v2 macro-F1 56.86% ± 0.90%</footer>
</main><script>
const examples={positive:["Microsoft","The latest update is excellent and everything works smoothly."],negative:["Microsoft","The update broke everything and the service is awful."],relevance:["AssassinsCreed","Vinland Saga looks amazing and I cannot wait to play it!"]};
document.querySelectorAll('[data-example]').forEach(b=>b.onclick=()=>{const v=examples[b.dataset.example];entity.value=v[0];text.value=v[1]});
analyze.onclick=async()=>{analyze.disabled=true;result.innerHTML='<div class="status">Analyzing…</div>';
try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({entity:entity.value,text:text.value})});const d=await r.json();if(!r.ok)throw Error(d.error||'Prediction failed');const pct=(d.confidence*100).toFixed(1);result.innerHTML=`<div class="label">${d.sentiment}</div><div class="confidence">Uncalibrated confidence · ${pct}%</div><div class="meter"><span style="width:${pct}%"></span></div>`}catch(e){result.innerHTML=`<div class="label">Unable to predict</div><div class="confidence"></div>`;result.querySelector('.confidence').textContent=e.message}finally{analyze.disabled=false}};
</script></body></html>"""


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def create_server(
    model_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """Create a local demo server and load the trusted model exactly once."""

    model = load_artifact(model_path)["model"]

    class DemoHandler(BaseHTTPRequestHandler):
        def _respond(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _respond_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._respond(status, _json_bytes(payload), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._respond(HTTPStatus.OK, DEMO_HTML.encode(), "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._respond_json(HTTPStatus.OK, {"status": "ok", "model_loaded": True})
            else:
                self._respond_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/predict":
                self._respond_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError("Request body size is invalid")
                payload = json.loads(self.rfile.read(length))
                entity = str(payload.get("entity", "")).strip()
                text = str(payload.get("text", "")).strip()
                if not entity or not text:
                    raise ValueError("Entity and text are required")
                if len(text) > MAX_TEXT_LENGTH:
                    raise ValueError(f"Text must be at most {MAX_TEXT_LENGTH} characters")
                inputs = pd.DataFrame({"entity": [entity], "text": [text]})
                probabilities = model.predict_proba(inputs)[0]
                index = int(probabilities.argmax())
                self._respond_json(
                    HTTPStatus.OK,
                    {
                        "entity": entity,
                        "text": text,
                        "sentiment": str(model.classes_[index]),
                        "confidence": float(probabilities[index]),
                        "confidence_is_calibrated": False,
                    },
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._respond_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), DemoHandler)


def demo_command(argv: Sequence[str] | None = None) -> None:
    """Start the local interactive demo."""

    parser = argparse.ArgumentParser(description="Serve the local sentiment demo.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/sentiment_model.joblib"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    server = create_server(args.model, host=args.host, port=args.port)
    print(f"Sentiment demo: http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
