FROM python:3.10

RUN apt-get update

COPY . /root/code/ptseries-sdk/

RUN \
	cd /root/code/ptseries-sdk && \
	pip install -e .[tests]
