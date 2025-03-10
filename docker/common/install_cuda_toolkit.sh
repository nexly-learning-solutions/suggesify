#!/bin/bash

set -ex

CUDA_VER="12.6.3_560.35.05"
CUDA_VER_SHORT="${CUDA_VER%_*}"

NVCC_VERSION_OUTPUT=$(nvcc --version)
OLD_CUDA_VER=$(echo $NVCC_VERSION_OUTPUT | grep -oP "\d+\.\d+" | head -n 1)
echo "The version of pre-installed CUDA is ${OLD_CUDA_VER}."

reinstall_centos_cuda() {
    yum -y update
    yum -y install epel-release
    yum remove -y "cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
    rm -rf /usr/local/cuda-${OLD_CUDA_VER}
    wget -q https://developer.download.nvidia.com/compute/cuda/${CUDA_VER_SHORT}/local_installers/cuda_${CUDA_VER}_linux.run
    sh cuda_${CUDA_VER}_linux.run --silent --override --toolkit
    rm -f cuda_${CUDA_VER}_linux.run
}

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  centos)
    echo "Reinstall CUDA for CentOS 7..."
    reinstall_centos_cuda
    ;;
  *)
    echo "Skip for other OS..."
    ;;
esac
