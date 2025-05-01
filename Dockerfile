FROM python:3.11-slim

# Install system dependencies for science/machine learning libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libglib2.0-0 libsm6 libxext6 libxrender-dev git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the full app (make sure your code structure is correct)
COPY . .

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "53053"]
