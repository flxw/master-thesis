ARG cuda_version=9.2
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

RUN apt-get update && apt-get install -y wget git bzip2 gcc vim
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O anaconda.sh &&\
    /bin/bash anaconda.sh -b -p /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh  /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate thesis" >> ~/.bashrc
RUN git clone https://github.com/flxw/master-thesis-code ~/master-thesis-code

WORKDIR /root/master-thesis-code
RUN /opt/conda/bin/conda create --name thesis --file thesis-environment.yml

SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/bin/activate thesis && pip install -r requirements.txt
ADD . logs/

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENTRYPOINT git pull && /bin/bash
