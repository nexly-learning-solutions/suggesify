Bootstrap: docker
From: nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

%environment
    export DEBIAN_FRONTEND=noninteractive
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/openmpi/lib:${LD_LIBRARY_PATH}
    export PATH=src/bin:/usr/local/openmpi/bin:${PATH}
    export PYTHONPATH=/usr/local/lib/python3.8/site-packages:${PYTHONPATH}
    export MY_CUSTOM_VAR="custom_value"
    export BUILD_TYPE="${BUILD_TYPE:-Release}"
    export USE_GPU="${USE_GPU:-1}"

%setup
    cp -r . "${SINGULARITY_ROOTFS}"

%post
    set -euxo pipefail

    # Logging
    LOGFILE=/var/log/singularity_build.log
    exec > >(tee -i $LOGFILE) 2>&1

    # Functions
    error_exit() {
        echo "Error: $1"
        exit 1
    }

    check_command() {
        command -v "$1" >/dev/null 2>&1 || error_exit "$1 not found!"
    }

    install_package() {
        local package="$1"
        if ! dpkg -s "$package" >/dev/null 2>&1; then
            echo "Installing package: $package"
            apt-get install -y "$package" || error_exit "Failed to install $package"
        else
            echo "Package $package already installed"
        fi
    }

    # Update and install essential packages
    echo "Updating package lists..."
    apt-get update || error_exit "apt-get update failed"

    echo "Installing essential packages..."
    install_package build-essential
    install_package libatlas-base-dev
    install_package pkg-config
    install_package python3
    install_package python3-pip
    install_package python3-venv
    install_package unzip
    install_package wget
    install_package cmake
    install_package libgtest-dev
    install_package libopenmpi-dev
    install_package libjsoncpp-dev
    install_package libhdf5-dev
    install_package zlib1g-dev
    install_package libnetcdf-dev
    install_package libnetcdf-c++4-dev
    install_package libssl-dev
    install_package libffi-dev
    install_package libbz2-dev
    install_package liblzma-dev
    install_package libboost-all-dev
    install_package libgoogle-glog-dev
    install_package git
    install_package sudo
    install_package vim
    install_package curl
    install_package lsb-release
    install_package gnupg
    install_package ninja-build
    install_package jq

    echo "Cleaning up APT cache..."
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

    # Install Google Test
    echo "Installing Google Test..."
    cd /usr/src/googletest || error_exit "Google Test directory not found"
    cmake . || error_exit "cmake failed"
    make || error_exit "make failed"
    cp *.a /usr/lib || error_exit "Copying Google Test libraries failed"

    # Add user and set up sudo
    echo "Adding user 'user'..."
    useradd -m -s /bin/bash user || error_exit "User addition failed"
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

    # Clone repositories
    echo "Cloning repositories..."
    mkdir -p /opt/3rdparty
    cd /opt/3rdparty || error_exit "Failed to change directory to /opt/3rdparty"
    declare -a repos=(
        "https://github.com/NVIDIA/cutlass.git"
        "https://github.com/nlohmann/json.git"
        "https://github.com/jarro2783/cxxopts.git"
        "https://github.com/NVIDIA/NVTX.git"
        "https://github.com/OpenMathLib/OpenBLAS.git"
        "https://github.com/pybind/pybind11.git"
        "https://github.com/Unidata/netcdf-cxx4.git"
        "https://github.com/NVIDIA/TensorRT.git"
        "https://github.com/open-mpi/ompi.git"
        "https://github.com/python/cpython.git"
        "https://github.com/NVIDIA/nccl.git"
        "https://github.com/pytorch/pytorch.git"
        "https://github.com/NVIDIA/cudnn-frontend.git"
        "https://github.com/open-source-parsers/jsoncpp.git"
        "https://github.com/google/highway.git"
        "https://github.com/google/googletest.git"
        "https://github.com/GerHobbelt/pthread-win32.git"
        "https://github.com/boostorg/boost.git"
        "https://github.com/google/glog.git"
        "https://github.com/NVIDIA/cccl.git"
    )
    for repo in "${repos[@]}"; do
        git clone "$repo" || error_exit "Cloning $repo failed"
    done
    cd cxxopts && git checkout v3.1.1 || error_exit "Checking out cxxopts v3.1.1 failed"

    # Install dependencies
    install_dependency() {
        local url="$1"
        local tarfile=$(basename "$url")
        local dir="${tarfile%.tar.gz}"
        echo "Downloading $tarfile..."
        cd /tmp || error_exit "Failed to change directory to /tmp"
        wget "$url" || error_exit "Downloading $tarfile failed"
        tar xzf "$tarfile" || error_exit "Extracting $tarfile failed"
        cd "$dir" || error_exit "Failed to change directory to $dir"
        ./configure --prefix=/usr/local || error_exit "Configuring $dir failed"
        make -j8 || error_exit "Building $dir failed"
        make install || error_exit "Installing $dir failed"
        rm -rf /tmp/*
    }

    # Install OpenMPI
    echo "Installing OpenMPI..."
    install_dependency "https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.1.tar.gz"

    # Install JsonCpp
    echo "Installing JsonCpp..."
    cd /tmp || error_exit "Failed to change directory to /tmp"
    wget https://github.com/open-source-parsers/jsoncpp/archive/svn-import.tar.gz || error_exit "Downloading JsonCpp failed"
    tar xzf svn-import.tar.gz || error_exit "Extracting JsonCpp failed"
    cd jsoncpp-svn-import || error_exit "Failed to change directory to JsonCpp source"
    mkdir -p build/release
    cd build/release || error_exit "Failed to change directory to JsonCpp build/release"
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DJSONCPP_LIB_BUILD_SHARED=OFF ../.. || error_exit "Configuring JsonCpp failed"
    make -j8 || error_exit "Building JsonCpp failed"
    make install || error_exit "Installing JsonCpp failed"
    rm -rf /tmp/*

    # Install HDF5
    echo "Installing HDF5..."
    install_dependency "ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.9.tar.gz"

    # Install Zlib
    echo "Installing Zlib..."
    install_dependency "https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4/zlib-1.2.8.tar.gz"

    # Install NetCDF
    echo "Installing NetCDF..."
    install_dependency "https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4.1.3.tar.gz"

    # Install NetCDF C++ Bindings
    echo "Installing NetCDF C++ Bindings..."
    install_dependency "https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-cxx4-4.2.tar.gz"

    # Python Environment Setup
    echo "Setting up Python virtual environment..."
    python3 -m venv /opt/venv || error_exit "Creating Python virtual environment failed"
    source /opt/venv/bin/activate
    pip install --upgrade pip || error_exit "Upgrading pip failed"
    pip install numpy scipy pandas matplotlib scikit-learn || error_exit "Python package installation failed"

    # Final configuration
    echo "Final configuration..."
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/openmpi/lib:${LD_LIBRARY_PATH}
    export PATH=/src/bin:/usr/local/openmpi/bin:/usr/local/bin:/usr/local/include/bin:/usr/local/cuda-12.5.1/bin:${PATH}

    echo "Installing..."
    cd /src || error_exit "Failed to change directory to /src"
    sudo -u user make install || error_exit "Installation of failed"

    # Clean up
    echo "Cleaning up..."
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

%runscript
    echo "Starting the container..."
    exec /bin/bash "$@"