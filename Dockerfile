FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Set the working directory
WORKDIR /code

# Copy dependency file and install packages
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire app folder
COPY ./app /code/app

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "53053"]
