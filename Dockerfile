FROM continuumio/miniconda3
WORKDIR /excut

COPY envs/excut_env_minimum.yml ./env.yml
#RUN apk add openssl
#RUN conda config --set ssl_verify no
ENV PATH /opt/conda/envs/excut/bin:$PATH
RUN conda env create -f env.yml &&  echo "source activate excut" > ~/.bashrc

COPY . /excut

#CMD ["python", "docker_test.py"]




