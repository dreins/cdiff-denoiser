FROM python:3.9-slim-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends \

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

RUN which python
RUN which pip
RUN which uvicorn

# Explicitly use the uvicorn executable.  This is the most robust approach.
CMD ["python3", "-m", "app.main"]
