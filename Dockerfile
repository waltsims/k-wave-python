FROM python:3.7.0-slim

RUN apt-get update
RUN apt-get install -y gfortran libopenblas-dev liblapack-dev libgl1-mesa-glx libglib2.0-0 libgl1-mesa-glx libglib2.0-0
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install pytest
