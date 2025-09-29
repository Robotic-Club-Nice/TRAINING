#!/bin/bash

set -e
echo "Installation de l'environnement Robocar avec TensorFlow 2.15..."

# Installer dépendances système
sudo apt update
sudo apt install -y python3-dev build-essential wget

# Supprimer ancienne installation Miniconda si elle existe
echo "Suppression de l'ancienne installation Miniconda..."
rm -rf ~/miniconda3

# Supprimer ancien fichier Miniconda téléchargé
echo "Suppression de l'ancien fichier Miniconda..."
rm -f Miniconda3-*.sh

# Télécharger et installer Miniconda
echo "Téléchargement de Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.4.0-0-Linux-x86_64.sh
echo "Installation de Miniconda..."
bash ./Miniconda3-py311_24.4.0-0-Linux-x86_64.sh

# Initialiser conda
source ~/miniconda3/bin/activate
conda init

# Créer environnement conda avec Python 3.11
echo "Création de l'environnement conda don36..."
conda create -n don36 python=3.11 -y

# Activer l'environnement
echo "Activation de l'environnement..."
conda activate don36

# Installer TensorFlow 2.15
echo "Installation de TensorFlow 2.15..."
pip install tensorflow==2.15

# Installer CUDA toolkit
echo "Installation de CUDA toolkit..."
conda install cudatoolkit=11 -c pytorch -y

# Installer les dépendances principales
echo "Installation des dépendances principales..."
pip install matplotlib
pip install kivy
pip install kivy-garden.matplotlib
pip install pandas
pip install plotly
pip install albumentations

# Installer les dépendances supplémentaires
echo "Installation des dépendances supplémentaires..."
pip install pillow
pip install docopt
pip install tornado
pip install requests
pip install PrettyTable
pip install paho-mqtt
pip install simple_pid
pip install progress
pip install pyfiglet
pip install psutil
pip install pynmea2
pip install pyserial
pip install utm
pip install pyyaml
pip install opencv-python

# Forcer la bonne version de NumPy APRÈS OpenCV
echo "Correction de la version NumPy..."
pip install "numpy<2.0.0,>=1.23.5" --force-reinstall

echo ""
echo "Vérification de l'installation..."
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import numpy as np; print('NumPy:', np.__version__)"
python -c "import PIL; print('PIL: OK')"

echo ""
echo "Installation terminée!"
echo "Pour utiliser l'environnement:"
echo "conda activate don36"
echo "pip install \"numpy<2.0.0,>=1.23.5\" --force-reinstall"
echo "python main.py --tub ./data"