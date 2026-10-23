# Public demo deployment

The public portfolio demo should serve the classical entity-aware model. It is not the
quality-leading model, but its approximately 3.04 MiB benchmark artifact is far smaller
than the 478.92 MiB Twitter-RoBERTa artifact and requires no GPU runtime.

## Build a self-contained image

Generate the trusted artifact locally from the reviewed pipeline:

```bash
uv sync --locked --extra dev
uv run sentiment-benchmark --jobs 2
```

Build the hardened base and then the release layer containing only that artifact:

```bash
docker build --tag sentiment-demo:base .
docker build --file Dockerfile.public \
  --build-arg BASE_IMAGE=sentiment-demo:base \
  --tag sentiment-demo:public .
```

`artifacts/sentiment_model.joblib` remains ignored by Git. `.dockerignore` permits only
that exact artifact into the release build context. Do not substitute an untrusted
joblib file because deserialization can execute code.

Verify the self-contained image without a host mount:

```bash
docker run --detach --name sentiment-demo-public \
  --publish 127.0.0.1:8000:8000 \
  --read-only --tmpfs /tmp:size=16m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges:true \
  sentiment-demo:public
curl --fail http://127.0.0.1:8000/healthz
```

CI performs the same no-volume health and inference test using a small fixture artifact.

## Hosting contract

Push the verified image to a private or public container registry and configure the
hosting service to:

- route HTTPS traffic to container port 8000;
- preserve the image's non-root user, read-only filesystem, and health check;
- allocate at least 1 GiB memory and one CPU for the classical scientific-Python stack;
- restart unhealthy instances and cap concurrent requests;
- apply provider-level request throttling because the application intentionally has no
  user accounts;
- avoid request-body logging so submitted text is not retained;
- expose `/healthz` for readiness and liveness checks;
- keep at least one instance warm only if the portfolio budget permits it.

The application validates request size and text length, sends a restrictive content
security policy, and does not persist input. Provider-managed TLS, abuse controls,
availability, logs, and spending limits remain deployment responsibilities.

## Release checklist

Before adding a live URL to the README:

1. Record the Git commit and model-evidence version used to build the image.
2. Run the local no-volume smoke test above.
3. Scan the final image and confirm the GitHub dependency audit is green.
4. Verify `/healthz` and a positive, negative, neutral, and irrelevant example through
   the public HTTPS endpoint.
5. Confirm no submitted text appears in provider logs.
6. Document that the public demo serves the classical model while Twitter-RoBERTa is the
   benchmark quality leader.
7. Set a budget or automatic shutdown policy; free hosting may sleep or disappear.
