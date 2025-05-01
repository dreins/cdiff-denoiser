FROM python:3.8-slim

# Set the working directory
WORKDIR /app

COPY . /app/

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements.txt
COPY requirements.txt /app/

# Install the Python dependencies
RUN pip install --upgrade -r requirements.txt

# Expose the port
EXPOSE 53053

# Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "53053"]
