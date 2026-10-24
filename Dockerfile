FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY LICENSE ./
COPY src/ src/

# Install with the [mcp] extra so the MCP HTTP transport mounts on startup.
RUN pip install --no-cache-dir '.[mcp]'

LABEL org.opencontainers.image.title="LibreWXR" \
      org.opencontainers.image.description="Self-hostable Rain Viewer API replacement - radar composites from public sources, served as a Rain Viewer-compatible JSON/tile API." \
      org.opencontainers.image.source="https://github.com/JoshuaKimsey/LibreWXR" \
      org.opencontainers.image.url="https://github.com/JoshuaKimsey/LibreWXR" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"

EXPOSE 8080

CMD ["python", "-m", "librewxr.main"]
