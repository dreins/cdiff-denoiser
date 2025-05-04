FROM python:3.9

WORKDIR /app
COPY . /app

# Show Python and pip version
RUN python --version && pip --version

# Show your requirements file content
RUN echo "==== requirements.txt ====" && cat requirements.txt

# Install requirements
RUN pip install -r requirements.txt

# Check if uvicorn installed correctly
RUN echo "==== uvicorn path and version ====" && which uvicorn && uvicorn --version

# List installed packages (to confirm everything)
RUN pip freeze

# Show what's in your app directory
RUN echo "==== app structure ====" && ls -R /app

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "53053"]
