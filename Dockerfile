FROM python:3.9

# Install Python 3
RUN apt-get update
RUN apt-get  install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y git

# Install Create Python alias for python3
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc


WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN which python
RUN which pip
RUN which uvicorn
