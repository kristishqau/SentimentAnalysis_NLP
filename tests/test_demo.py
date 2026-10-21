from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from sentiment_analysis.artifacts import save_artifact
from sentiment_analysis.demo import create_server
from sentiment_analysis.modeling import MODEL_INPUT_COLUMNS, build_pipeline


def test_demo_health_prediction_and_validation(sample_frame, tmp_path: Path) -> None:
    model = build_pipeline(max_features=100)
    model.fit(sample_frame[MODEL_INPUT_COLUMNS], sample_frame["sentiment"])
    model_path = tmp_path / "model.joblib"
    save_artifact(model_path, model=model, metadata={"purpose": "demo-test"})
    server = create_server(model_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base_url}/healthz") as response:
            assert json.load(response) == {"status": "ok", "model_loaded": True}

        request = urllib.request.Request(
            f"{base_url}/api/predict",
            data=json.dumps(
                {"entity": "ExampleEntity", "text": "excellent delightful amazing"}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            prediction = json.load(response)
        assert prediction["sentiment"] == "Positive"
        assert prediction["confidence_is_calibrated"] is False

        invalid = urllib.request.Request(
            f"{base_url}/api/predict",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(invalid)
        except urllib.error.HTTPError as error:
            assert error.code == 400
            assert "required" in json.load(error)["error"]
        else:
            raise AssertionError("Invalid prediction input should fail")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
