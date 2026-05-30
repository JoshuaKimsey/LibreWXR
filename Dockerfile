FROM python:3.12-slim

WORKDIR /app

# libfftw3-3 is the shared library pyfftw binds to; pysteps S-PROG uses
# pyfftw for the spectral cascade FFTs when available and falls back to
# numpy FFT otherwise (~2-3× slower).  The apt package is ~1 MB.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libfftw3-3 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["python", "-m", "librewxr.main"]
