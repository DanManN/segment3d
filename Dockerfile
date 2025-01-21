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
RUN chown -R user:user /home/user

# setup environment
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN apt update && apt upgrade curl wget git tmux -y \
	&& rm -rf /var/lib/apt/lists/*

# setup conda
ENV PATH=/home/user/miniconda3/bin:$PATH
RUN mkdir -p /home/user/miniconda3 && \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/user/miniconda3/miniconda.sh && \
	bash /home/user/miniconda3/miniconda.sh -b -u -p /home/user/miniconda3 && \
	rm -rf /home/user/miniconda3/miniconda.sh

USER user
SHELL ["/usr/bin/bash", "-ic"]

RUN conda create -n ovir3d python=3.10 && conda init bash
RUN conda create -n langsam python=3.11

# ENV CUDA_HOME=/usr/local/cuda-11.4
#RUN conda activate ovir3d && \
#	conda install -y nvidia::cuda-toolkit
RUN conda activate ovir3d && conda install libffi==3.3
RUN conda activate ovir3d && \
	pip3 install rospkg catkin-pkg open3d scikit-image scikit-learn torchmetrics cupy-cuda11x opencv-python matplotlib pyzmq
RUN conda activate langsam && \
	pip install torch==2.4.1 torchvision==0.19.1
RUN conda activate langsam && \
	pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
RUN conda activate langsam && \
	python -c "from lang_sam import LangSAMLangSAM()"

RUN conda activate ovir3d && \
	pip uninstall em && \
	pip install empy==3.3.4

# Installing catkin package
RUN mkdir -p /home/user/workspace
COPY --chown=user . /home/user/workspace/src/perception/
RUN sudo chown -R user:user /home/user/workspace
RUN source /opt/ros/noetic/setup.bash && \
	conda activate ovir3d && \
	cd /home/user/workspace && catkin_make

# update bashrc
RUN echo "source ~/workspace/devel/setup.bash" >> ~/.bashrc && \
	echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc && \
	echo "conda activate ovir3d" >> ~/.bashrc

CMD ["bash"]
