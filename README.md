# Bag of Words Text Classifier

A simple neural network with embedding layer, bias, and xavier initialization

## Usage

Build the docker image with:

`docker build -t bow_text_classifier .`

from the repo root.

To use the trained model for inference, pass the input sentence as follows:

`docker run -it --gpus bow_text_classifier "this is a test sentence"`

Output class will be printed to STDOUT.

## Training

Please refer to `notebooks/model_demo.ipynb` for training instructions.