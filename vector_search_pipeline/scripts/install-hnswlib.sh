#!/usr/bin/env bash

DIRECTORY_NAME="hnswlib"
HNSW_VERSION="v0.8.0"
BUILD_DIR="build"

# download hnsw github and checkout to version number
if [[ -n "${DIRECTORY_NAME}" ]]; then
    git clone https://github.com/nmslib/hnswlib.git &>/dev/null
    git checkout "${HNSW_VERSION}" &>/dev/null
fi
cd hnswlib || return

# setup install prefix
echo "here ${HNSWLIB_INSTALL_PREFIX}"
if [ -z "${HNSWLIB_INSTALL_PREFIX}" ]; then
    install_prefix="/usr/local"
else
    install_prefix=$HNSWLIB_INSTALL_PREFIX
fi
echo "Hnsw library will be installed in $install_prefix"

read -p "Install hnswlib Python interface? (y/n): " response
response=${response,,}

if [[ "$response" != "y" && "$response" != "n" ]]; then
    echo "Invalid response. Please enter 'y' or 'n'."
    return
fi

# delete previous cpp installation
rm -rf "${install_prefix}"/include/hnswlib
rm -rf "${install_prefix}"/lib/cmake/hnswlib

# install cpp
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" || return
cmake .. -DCMAKE_INSTALL_PREFIX="${install_prefix}"
make -j"$(nproc)"
make install
cd ..

if [[ "$response" == "y" ]]; then
    echo "Installing hnsw Python interface..."
    pip install --user .
fi

echo "FAISS installed successfully."
cd ..
rm -rf "${DIRECTORY_NAME}"
