#!/bin/bash

curl -O https://curl.se/download/curl-8.10.1.tar.gz
tar -xzf curl-8.10.1.tar.gz
rm curl-8.10.1.tar.gz
cd curl-8.10.1
if [ -z $CURL_INSTALL_PREFIX ]; then
    install_prefix="/usr/local"
else
    install_prefix=$CURL_INSTALL_PREFIX
fi
echo "CURL library will be installed in $install_prefix"

./configure  --prefix="${install_prefix}" --with-openssl  --without-libpsl
NPROC=`nproc`
if [ $NPROC -lt 2 ]; then
    NPROC=2
fi

make -j `expr $NPROC - 1` 2>err.log
if [ $? -ne 0 ]; then
    echo "Error during build. Check err.log for details."
    exit 1
fi
make install
cd ..
rm -rf curl-8.10.1
