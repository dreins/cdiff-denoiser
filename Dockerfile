# Use the official Python image as the base image
FROM python:3.11-slim

# Set environment variables to ensure non-interactive installation of packages
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install the Python dependencies from the requirements.txt
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . /app/

# Expose the port the app will run on
EXPOSE 53053

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "53053"]
