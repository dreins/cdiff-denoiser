FROM python:3.9-slim-buster

# Install system dependencies (less is usually better in Docker)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \

# Don't create aliases in the Dockerfile.  It's better to be explicit.
# WORKDIR /app  # Correct WORKDIR is set later, so don't set it here.

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt # Install requirements
COPY . /app

RUN which python
RUN which pip
RUN which uvicorn

# Explicitly use the uvicorn executable.  This is the most robust approach.
CMD ["python3", "-m", "app.main"]
