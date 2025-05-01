FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements.txt
COPY requirements.txt /app/

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Debug step: Ensure `uvicorn` is installed
RUN which uvicorn || echo "Uvicorn not found"

# Copy the rest of the application code
COPY . /app/

# Expose the port
EXPOSE 53053

# Set the command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "53053"]
