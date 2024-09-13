FROM mzahana/ros-noetic-cuda11.4.2

# set user permissions
#RUN useradd user && \
RUN echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
	chmod 0440 /etc/sudoers.d/user && \
	mkdir -p /home/user && \
	chown user:user /home/user && \
	chsh -s /bin/bash user
RUN echo 'root:root' | chpasswd
RUN echo 'user:user' | chpasswd

# setup environment
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN apt update && apt upgrade curl wget git -y

# setup conda
ENV PATH=/home/user/miniconda3/bin:$PATH
RUN mkdir -p /home/user/miniconda3 && \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/user/miniconda3/miniconda.sh && \
	bash /home/user/miniconda3/miniconda.sh -b -u -p /home/user/miniconda3 && \
	rm -rf /home/user/miniconda3/miniconda.sh

USER user
SHELL ["/usr/bin/bash", "-ic"]

RUN conda create -n ovir3d python=3.10 && conda init bash
RUN conda activate ovir3d && conda install libffi==3.3

RUN conda activate ovir3d && \
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
RUN conda activate ovir3d && \
	pip3 uninstall em && \
	pip3 install mss lvis empy==3.3.4 nltk ftfy timm regex numba \
	rospkg catkin-pkg \
	open3d einops fasttext GitPython dataclasses \
	scikit-image scikit-learn torchmetrics cupy-cuda11x opencv-python \
	git+https://github.com/openai/CLIP.git \
	git+https://github.com/facebookresearch/detectron2.git

RUN conda activate ovir3d && \
	python -c "import clip, torchclip.load('ViT-B/32', 'cuda' if torch.cuda.is_available() else 'cpu')"

RUN mkdir -p /home/user/workspace/src/perception
COPY --chown=user ./requirements.txt /home/user/workspace/src/perception/
ENV CUDA_HOME=/usr/local/cuda-11.4
RUN conda activate ovir3d && \
	cd /home/user/workspace/src/perception && \
	pip3 install -r requirements.txt && \
	pip3 install --no-deps git+https://github.com/luca-medeiros/lang-segment-anything.git

RUN conda activate ovir3d && \
	python -c "from lang_sam import LangSAMLangSAM('vit_b')"

# Installing catkin package
COPY --chown=user . /home/user/workspace/src/perception/
RUN source /opt/ros/noetic/setup.bash && \
	conda activate ovir3d && \
	cd /home/user/workspace && catkin_make

# update bashrc
RUN echo "source ~/workspace/devel/setup.bash" >> ~/.bashrc && \
	echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc && \
	echo "conda activate ovir3d" >> ~/.bashrc

CMD ["bash"]
