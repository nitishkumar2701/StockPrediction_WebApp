#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies for numpy
apt-get update
apt-get install -y python3-dev

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
