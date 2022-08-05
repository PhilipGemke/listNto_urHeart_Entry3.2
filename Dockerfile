FROM python:3.10.1-buster

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER philip.gemke@tu-braunschweig.de

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
RUN apt-get -y update
RUN apt-get install -y libsndfile1


## Include the following line if you have a requirements.txt file.

RUN pip install -r requirements.txt
