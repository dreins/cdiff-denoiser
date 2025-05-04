FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Set the working directory to /code
WORKDIR /code

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app directory into /code/app inside the container
COPY ./app /code/app

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "53053"]
