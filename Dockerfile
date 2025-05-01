FROM python:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

# Copy the application code into the container
COPY . .

# Expose the port for the app
EXPOSE 53053

# Command to run the app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "53053"]
