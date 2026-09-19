FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml README.md ./
COPY eeg_decode ./eeg_decode
COPY frontend ./frontend
COPY examples ./examples
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
ENV EEG_HOST=0.0.0.0 EEG_PORT=8000
CMD ["python", "-m", "eeg_decode"]
