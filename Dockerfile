FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/openmpi/bin/:/opt/bin/:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/openmpi/lib/:${LD_LIBRARY_PATH}

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libatlas-base-dev \
        pkg-config \
        python3 \
        unzip \
        wget \
        cmake \
        libgtest-dev \
        libopenmpi-dev \
        libjsoncpp-dev \
        libhdf5-dev \
        zlib1g-dev \
        libnetcdf-dev \
        libnetcdf-c++4-dev \
        libssl-dev \
        libffi-dev \
        libbz2-dev \
        liblzma-dev \
        libboost-all-dev \
        libgoogle-glog-dev \
        git \
        curl \
    && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p /opt/3rdparty

WORKDIR /opt/3rdparty

RUN git clone https://github.com/OpenMathLib/OpenBLAS.git && \
    cd OpenBLAS && \
    ./configure --prefix=/usr/local --enable-shared && \
    make -j $(nproc) && \
    make install && \
    cd ..

RUN git clone https://github.com/NVIDIA/TensorRT.git && \
    cd TensorRT && \
    ./configure --prefix=/usr/local --enable-shared && \
    make -j $(nproc) && \
    make install && \
    cd ..

RUN git clone https://github.com/open-mpi/ompi.git && \
    cd ompi && \
    ./configure --prefix=/usr/local --enable-shared && \
    make -j $(nproc) && \
    make install && \
    cd ..

RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    ./configure --prefix=/usr/local --enable-shared && \
    make -j $(nproc) && \
    make install && \
    cd ..

RUN git clone https://github.com/NVIDIA/cccl.git && \
    cd cccl && \
    ./configure --prefix=/usr/local --enable-shared && \
    make -j $(nproc) && \
    make install && \
    cd ..

RUN git clone https://github.com/boostorg/boost.git && \
    cd boost && \
    ./bootstrap.sh && \
    ./b2 install && \
    cd ..

RUN git clone https://github.com/google/glog.git && \
    cd glog && \
    cmake . && \
    make -j $(nproc) && \
    make install && \
    cd ..

RUN git clone https://github.com/pybind/pybind11.git

RUN git clone https://github.com/Unidata/netcdf-cxx4.git && \
    cd netcdf-cxx4 && \
    cmake . -DCMAKE_INSTALL_PREFIX=/usr/local -DUSE_NETCDF=ON -DBUILD_TESTING=OFF && \
    make -j $(nproc) install && \
    cd ..

RUN git clone https://github.com/python/cpython.git && \
    cd cpython && \
    ./configure --prefix=/usr/local --enable-shared && \
    make -j $(nproc) && \
    make install && \
    cd ..

RUN git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    ./configure --prefix=/usr/local --enable-shared && \
    make -j $(nproc) install && \
    cd ..

RUN git clone https://github.com/NVIDIA/cudnn-frontend.git && \
    cd cudnn-frontend && \
    ./configure --prefix=/usr/local --enable-shared && \
    make -j $(nproc) install && \
    cd ..

RUN git clone https://github.com/open-source-parsers/jsoncpp.git && \
    cd jsoncpp && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON && \
    make -j $(nproc) install && \
    cd ../..

RUN git clone https://github.com/google/highway.git && \
    cd highway && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON && \
    make -j $(nproc) install && \
    cd ../..

RUN git clone https://github.com/google/googletest.git && \
    cd googletest && \
    cmake . && \
    make -j $(nproc) && \
    make install && \
    cd ..

RUN git clone https://github.com/NVIDIA/cutlass.git && \
    cd cutlass && \
    cmake . -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON && \
    make -j $(nproc) install && \
    cd ..

RUN git clone https://github.com/GerHobbelt/pthread-win32.git && \
    cd pthread-win32 && \
    cmake . && \
    make -j $(nproc) install && \
    cd ..

FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

COPY --from=builder /usr/local /usr/local

RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*