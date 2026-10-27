# Live demo image: FastAPI + MCP + A2A + web UI, using the deterministic mock model by default.
# Compatible with Hugging Face Docker Spaces (port 7860, non-root user).
FROM python:3.12-slim

RUN useradd -m -u 1000 app
WORKDIR /home/app/agent-platform
ENV PATH="/home/app/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    CREWAI_DISABLE_TELEMETRY=true \
    OTEL_SDK_DISABLED=true

COPY --chown=app requirements-frameworks.txt ./
USER app
RUN pip install --no-cache-dir -r requirements-frameworks.txt

COPY --chown=app pyproject.toml README.md LICENSE ./
COPY --chown=app src ./src
RUN pip install --no-cache-dir --no-deps .

EXPOSE 7860
# Gemini mode stays off unless ENABLE_GEMINI_DEMO and GEMINI_API_KEY are set as server secrets.
CMD ["agent-platform", "serve", "--host", "0.0.0.0", "--port", "7860"]
