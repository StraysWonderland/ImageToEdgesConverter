# ImageToEdgesConverter

A simple tool for detecting edges in a set images of a given dataset and storing a new image containing only the edges for each image in the set.
Useful for generating a dataset for training a GAN for image-to-image-translation

### Requirements:
- python 3.7
- open cv
- matplotlib
- numpy

These requirements in the requirements.txt.
To install them, simply open a terminal in the directory and run

    pip install -r requirements.txt
    
## Usage
open a terminal in the directory of the converter and type

    python ImageToEdgesConverter.py [--i] [path to target image]


