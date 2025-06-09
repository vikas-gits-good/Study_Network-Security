FROM python:3.10-alpine

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Update apt, install awscli, clean cache in one RUN to reduce layers and size
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD ["python3", "app.py"]