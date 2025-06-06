#!/bin/bash

set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

set_bash_env() {
  if [ ! -f ${BASH_ENV} ];then
    touch ${BASH_ENV}
  fi
  if [ ! -f ${ENV} ];then
    touch ${ENV}
    (echo "test -f ${ENV} && source ${ENV}" && cat ${BASH_ENV}) > /tmp/shinit_f
    mv /tmp/shinit_f ${BASH_ENV}
  fi
}

init_ubuntu() {
  apt-get update
  apt-get install -y --no-install-recommends \
    ccache \
    gdb \
    git-lfs \
    libffi-dev \
    python3-dev \
    python3-pip \
    python-is-python3 \
    wget \
    pigz
  if ! command -v mpirun &> /dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev
  fi
  apt-get clean
  rm -rf /var/lib/apt/lists/*
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> "${ENV}"
  if [[ $(apt list --installed | grep libnvinfer) ]]; then
    apt-get remove --purge -y libnvinfer*
  fi
  if [[ $(apt list --installed | grep tensorrt) ]]; then
    apt-get remove --purge -y tensorrt*
  fi
  pip3 uninstall -y tensorrt
}

install_gcc_centos() {
  yum -y update
  GCC_VERSION="9.5.0"
  yum install -y gcc gcc-c++ file libtool make wget bzip2 bison yacc flex
  wget -q ${GITHUB_URL}/gcc-mirror/gcc/archive/refs/tags/releases/gcc-${GCC_VERSION}.tar.gz -O /tmp/gcc-${GCC_VERSION}.tar.gz
  tar -xf /tmp/gcc-${GCC_VERSION}.tar.gz -C /tmp/ && cd /tmp/gcc-releases-gcc-${GCC_VERSION}
  ./contrib/download_prerequisites
  ./configure --disable-multilib --enable-languages=c,c++ --with-pi
  make -j$(nproc) && make install
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64' >> "${ENV}"
  cd .. && rm -rf /tmp/gcc-*
  yum clean all
}

init_centos_mirror() {
  sed -i -e 's/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-*
  sed -E -i -e 's/#baseurl=http:\/\/mirror.centos.org\/centos\/(\$releasever|7)\/([[:alnum:]_-]*)\/\$basearch\//baseurl=https:\/\/vault.centos.org\/7.9.2009\/\2\/\$basearch\//g' /etc/yum.repos.d/CentOS-*
}

install_python_centos() {
  PYTHON_VERSION=$1
  PYTHON_MAJOR="3"
  PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
  PYTHON_ENV_FILE="/tmp/python${PYTHON_VERSION}_env"
  yum -y update
  yum -y install centos-release-scl-rh epel-release
  init_centos_mirror
  cat "/etc/yum.repos.d/CentOS-SCLo-scl-rh.repo"
  yum-builddep -y python3 && yum remove -y python3
  yum install -y wget yum-utils make gcc openssl11 openssl11-devel bzip2-devel libffi-devel zlib-devel
  ln -sf /usr/lib64/pkgconfig/openssl11.pc /usr/lib64/pkgconfig/openssl.pc
  curl -L ${PYTHON_URL} | tar -zx -C /tmp
  cd /tmp/Python-${PYTHON_VERSION}
  bash -c "./configure --enable-shared --prefix=/opt/python/${PYTHON_VERSION} --enable-ipv6 \
    LDFLAGS=-Wl,-rpath=/opt/python/${PYTHON_VERSION}/lib,--disable-new-dtags && make -j$(nproc) && make install"
  ln -s /opt/python/${PYTHON_VERSION}/bin/python3 /usr/local/bin/python
  echo "export PATH=\$PATH:/opt/python/${PYTHON_VERSION}/bin" >> "${PYTHON_ENV_FILE}"
  echo "source ${PYTHON_ENV_FILE}" >> "${ENV}"
  yum clean all
  cd .. && rm -rf /tmp/Python-${PYTHON_VERSION}
}

install_pyp_centos() {
  bash -c "pip3 install 'urllib3<2.0' pytest"
}

install_devtoolset_centos() {
  DEVTOOLSET_ENV_FILE="/tmp/devtoolset_env"
  yum -y update
  yum -y install centos-release-scl-rh epel-release
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> "${ENV}"
  CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
  YUM_CUDA=${CUDA_VERSION/./-}
  yum -y install vim wget git-lfs rh-git227 devtoolset-10 libffi-devel
  yum -y install openmpi3 openmpi3-devel pigz
  echo "source scl_source enable rh-git227" >> "${ENV}"
  echo "source scl_source enable devtoolset-10" >> "${DEVTOOLSET_ENV_FILE}"
  echo "source ${DEVTOOLSET_ENV_FILE}" >> "${ENV}"
  echo 'export PATH=$PATH:/usr/lib64/openmpi3/bin' >> "${ENV}"
  yum clean all
}

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
set_bash_env
case "$ID" in
  ubuntu)
    init_ubuntu
    ;;
  centos)
    init_centos_mirror
    install_python_centos $1
    install_pyp_centos
    install_gcc_centos
    install_devtoolset_centos
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
