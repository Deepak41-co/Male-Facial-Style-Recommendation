# --- Base Image ---
# CHANGED FROM python:3.10-slim to python:3.10 to fix the mediapipe dependency issue.
# The 'slim' image often lacks the underlying system libraries (GLIBC) 
# needed for complex pre-compiled packages like MediaPipe.
FROM python:3.10

# --- System Dependencies for OpenCV & Mediapipe ---
# These system libraries are still necessary for the runtime of
# visual packages like opencv-python-headless and mediapipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
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
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT} augmented:app"