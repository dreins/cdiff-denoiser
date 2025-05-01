FROM continuumio/miniconda3

# Set working directory
WORKDIR /code

# Copy environment file and app
COPY environment.yaml /code/environment.yaml
COPY ./app /code/app

# # Clean conda cache (optional but useful)
# RUN conda clean --all

# Create conda environment
RUN conda env create -f /code/environment.yaml

# Use conda run to activate the environment for subsequent commands
SHELL ["conda", "run", "-n", "cold-diffusion", "/bin/bash", "-c"]

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "53053"]
