FROM python:3.9.0-slim

RUN apt-get update
RUN apt-get install -y gfortran libopenblas-dev liblapack-dev libgl1-mesa-glx libglib2.0-0 libgl1-mesa-glx libglib2.0-0 git
RUN pip install --upgrade pip
COPY pyproject.toml .
#COPY README.md .
COPY docs/ docs
COPY LICENSE .
COPY kwave/ kwave
RUN pip install '.[test]'
