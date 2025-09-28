#!/bin/bash

# Perbarui daftar paket
apt-get update

# Instal Python 3.9 dan dependensinya
# Python 3.9 masih menyertakan distutils sebagai modul terpisah
apt-get install -y python3.9 python3.9-dev python3.9-distutils

# Atur Python 3.9 sebagai default 'python3'
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

echo "Setup complete. Proceeding with pip installation."