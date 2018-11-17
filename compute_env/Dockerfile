FROM continuumio/miniconda3
RUN conda install numpy pandas seaborn scipy h5py
RUN conda install -c anaconda tensorflow-gpu
RUN conda install -c conda-forge scikit-image
RUN conda install -c conda-forge nibabel 
RUN conda install -c owlas transforms3d
RUN conda install -c conda-forge nipype
RUN conda install jupyter

# add user to execute commands
RUN useradd -ms /bin/bash mluser

USER mluser
ENV HOME /home/mluser
WORKDIR /home/mluser

COPY . /home/mluser

#CMD ["/bin/bash", "-c", "tensorboard --logdir=/home/mluser/notebooks/logs &"]
