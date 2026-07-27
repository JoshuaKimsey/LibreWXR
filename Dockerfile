FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Install with the [mcp] extra so the MCP HTTP transport mounts on startup.
RUN pip install --no-cache-dir '.[mcp]'

EXPOSE 8080

CMD ["python", "-m", "librewxr.main"]
