# --- Base Image ---
FROM python:3.10-slim

# --- System Dependencies for OpenCV & Mediapipe ---
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# --- Working Directory ---
WORKDIR /app

# --- Copy Backend Files ---
COPY Backend/ /app/

# --- Copy Frontend Files ---
COPY Frontend/ /app/Frontend/

# --- Copy Dependencies ---
COPY requirements.txt .

# --- Install Dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Expose Render Port ---
EXPOSE 10000

# --- Environment Variables ---
ENV PORT=10000
ENV PYTHONUNBUFFERED=1

# --- Start Command ---
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "augmented:app"]
