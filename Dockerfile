FROM nvidia/cuda:11.5.0-cudnn8-devel-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    apt-utils \
    bzip2 \
    libx11-6 \
    python \
    python3-pip \
    python3-venv \
    nano \
    zip \
    unzip \
    # lshw \
    wget \
    # ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y

RUN mkdir /home/venv/
RUN mkdir /home/Code

RUN cd /home/Code

RUN python3.8 -m venv /home/venv/python38_torch
RUN . /home/venv/python38_torch/bin/activate && pip install \
    numpy h5py torch scikit-image torchvision torchaudio torchsummary \
    pandas dill scipy ncls orjson tqdm matplotlib seaborn tensorboardX tensorboard \
    opencv-python glob2 pyyaml easydict pytorch-lightning optuna ezdxf plotly \
    glob2 scikit-learn kaleido EasyDict wandb albumentations \
    --extra-index-url https://download.pytorch.org/whl/cu115

RUN echo "alias activate_venv='. /home/venv/python38_torch/bin/activate'" >> ~/.bashrc
RUN echo "alias finish_container_setup='apt-get install lshw ffmpeg'" >> ~/.bashrc
RUN echo "alias nano_bashrc='nano ~/.bashrc'" >> ~/.bashrc
RUN echo "alias source_bashrc='source ~/.bashrc'" >> ~/.bashrc

# change the color prompt of the container
RUN echo "PS1='\${debian_chroot:+(\$debian_chroot)}\[\033[01;33m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$'"
# RUN source ~/.bashrc
# RUN cd /home/Code

WORKDIR /home/Code

RUN echo 'Please run finish_container_setup to finish the installation!!!'

# Set up the Conda environment
# ENV CONDA_AUTO_UPDATE_CONDA=false \
#     PATH=/home/user/miniconda/bin:$PATH
# COPY environment.yml /app/environment.yml
# RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
#  && chmod +x ~/miniconda.sh \
#  && ~/miniconda.sh -b -p ~/miniconda \
#  && rm ~/miniconda.sh \
#  && conda env update -n base -f /app/environment.yml \
#  && rm /app/environment.yml \
#  && conda clean -ya
