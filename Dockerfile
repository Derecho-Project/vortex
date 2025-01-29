FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

RUN apt update && \ 
apt install -y --no-install-recommends python3.10 python3-pip libgflags-dev vim nano tmux libreadline-dev swig libopenblas-dev git curl wget sudo build-essential automake libtool m4 autoconf && \ 
rm -rf /var/cache/apt/archives /var/lib/apt/lists

# install cmake version 3.30 for x86
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.6/cmake-3.30.6-linux-x86_64.tar.gz && \ 
tar -xzvf cmake-3.30.6-linux-x86_64.tar.gz && \ 
cd cmake-3.30.6-linux-x86_64 && \ 
mv bin/* /usr/local/bin/ && \ 
mv man/* /usr/local/share/man/ && \ 
mv share/* /usr/local/share/ && \ 
mv doc/* /usr/local/doc/ && \ 
cd .. && \ 
rm cmake-3.30.6-linux-x86_64.tar.gz && \ 
rm -rf cmake-3.30.6-linux-x86_64

# create user called vortexuser and set environment variables
# for derecho, cascade, and vortex

RUN adduser --disabled-password --gecos "" vortexuser
RUN adduser vortexuser sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >>/etc/sudoers

USER vortexuser
RUN mkdir /home/vortexuser/opt-dev /home/vortexuser/workspace

ENV OPT_HOME=/home/vortexuser/opt-dev
ENV CASCADE_INSTALL_PREFIX=${OPT_HOME}
ENV DERECHO_INSTALL_PREFIX=${OPT_HOME}
ENV VORTEX_INSTALL_PREFIX=${OPT_HOME}
ENV FAISS_INSTALL_PREFIX=${OPT_HOME}
ENV CURL_INSTALL_PREFIX=${OPT_HOME}/
ENV HNSWLIB_INSTALL_PREFIX=${OPT_HOME}
ENV SPDLOG_PREFIX=${OPT_HOME}
ENV CMAKE_PREFIX_PATH=${OPT_HOME}
ENV C_INCLUDE_PATH=${OPT_HOME}/include
ENV CPLUS_INCLUDE_PATH=${OPT_HOME}/include
ENV LIBRARY_PATH=${OPT_HOME}/lib
ENV LD_LIBRARY_PATH=${OPT_HOME}/lib
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDAToolKitRoot=/usr/local/cuda-12.3
ENV PYTHONPATH=$PYTHONPATH:${OPT_HOME}/lib
ENV PYTHONPATH=$PYTHONPATH:${OPT_HOME}/bin
ENV PATH=$PATH:${OPT_HOME}/bin
ENV PATH=$PATH:/home/vortexuser/.local/bin
ENV PATH=/usr/local/cuda/bin:$PATH

RUN pip3 install --user pip==24.0 pybind11==2.12.0 build virtualenv numpy==1.26.4

### Install derecho
WORKDIR /home/vortexuser/workspace
# Install prerequisites
RUN git clone https://github.com/Derecho-Project/derecho.git
RUN ./derecho/scripts/prerequisites/install-json.sh ~/opt-dev
RUN ./derecho/scripts/prerequisites/install-libfabric.sh ~/opt-dev
RUN ./derecho/scripts/prerequisites/install-mutils.sh ~/opt-dev
RUN ./derecho/scripts/prerequisites/install-mutils-containers.sh ~/opt-dev
RUN wget https://raw.githubusercontent.com/aliciayuting/vortextSetupScripts/refs/heads/main/spdlog_install.sh && \ 
chmod +x spdlog_install.sh && \ 
./spdlog_install.sh && \ 
rm spdlog_install.sh
# Build and install derecho
RUN sudo apt update && sudo apt install -y --no-install-recommends libssl-dev
RUN cd derecho && ./build.sh Release && cd build-Release && make install

### Install cascade
# Install prerequisites
RUN git clone https://github.com/Derecho-Project/cascade.git
RUN sudo apt install -y libboost-all-dev ragel --no-install-recommends
RUN ./cascade/scripts/prerequisites/install-hyperscan.sh ~/opt-dev
RUN ./cascade/scripts/prerequisites/install-libwsong.sh ~/opt-dev
RUN ./cascade/scripts/prerequisites/install-rpclib.sh ~/opt-dev
# build and install cascade
RUN cd cascade && ./build.sh Release && cd build-Release && make install
RUN cd ~/workspace/cascade/build-Release/src/service/python/dist && \ 
pip install --user derecho.cascade-1.0.2-py3-none-any.whl

### Install vortex
RUN cd ~/workspace && \ 
git clone https://github.com/Derecho-Project/vortex
RUN chmod +x -R ./vortex/scripts
RUN ./vortex/scripts/install-curl.sh
# install-faiss might fail because of non-compatible GPU arch
RUN echo "n" | ./vortex/scripts/install-faiss.sh
RUN echo "n" | ./vortex/scripts/install-hnswlib.sh

# bind mount host vortex repo
RUN rm -rf vortex

