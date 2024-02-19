FROM nvcr.io/nvidia/pytorch:23.10-py3

#RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $MY_PACKAGE

# install dependencies
RUN apt-get install git
RUN git clone https://github.com/Lightning-AI/lit-gpt /tmp/lit-gpt

# Creare una copia della curr folder su /tmp/ del container
#COPY . /tmp/

# Settare la working directory del container (dove copieremo il progetto)
WORKDIR /tmp/lit-gpt
