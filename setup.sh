#!/bin/bash

# Perbarui daftar paket
apt-get update

# Instal Python 3.10, developernya, dan distutils
apt-get install -y python3.10 python3.10-dev python3.10-distutils

# Atur Python 3.10 sebagai default 'python3'
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

echo "Setup complete. Proceeding with pip installation."