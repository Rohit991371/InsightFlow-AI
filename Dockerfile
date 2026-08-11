# ------ InsightFlow-AI . Docker Image -----
FROM python:3.12-slim

# Keep Python output unbuffered and skip .pyc files (nicer container logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1


WORKDIR /app


# System libs needed at runtime by matplotlib/seaborn/pillow/reportlab,
# plus curl for the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfreetype6 \
        libpng16-16 \
        fontconfig \
        libjpeg62-turbo \
        curl \
    && rm -rf /var/lib/apt/lists/*


# Install deps first so this layer caches across code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
    

# Now copy the actual app
COPY . .

# Folders the app writes to at runtime (also mounted as volumes in compose)
RUN mkdir -p uploads charts reports

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]