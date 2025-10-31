# --- Base Image ---
FROM python:3.10-slim-bullseye

# --- System Dependencies for OpenCV & Mediapipe ---
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# --- Set Working Directory ---
WORKDIR /app

# --- Copy Application Files ---
COPY Backend/ /app/
COPY Frontend/ /app/Frontend/
COPY requirements.txt .

# --- Install Dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Expose Port (for Render or Local use) ---
EXPOSE 10000

# --- Environment Variables ---
ENV PORT=10000
ENV PYTHONUNBUFFERED=1

# --- Start Gunicorn Server ---
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "augmented:app"]
