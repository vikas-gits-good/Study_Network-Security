FROM python:3.10-slim-bookworm

WORKDIR /app

# Copy the rest of the application code
COPY . .

# Update apt, install awscli, clean cache in one RUN to reduce layers and size
RUN apt update && \
    apt install -y --no-install-recommends awscli && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]