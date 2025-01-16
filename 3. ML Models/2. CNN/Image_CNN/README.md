## Image CNN Pipeline 

This folder contains scripts for a pretrained CNN model to classify input images. In particular,
the images represent the waterfall plots of mass spectrometry data—from commercial and Mars instruments—
to classify the chemical signatures present. This specific problem falls under the 
multi-label multi-class classification.

The code is modified from https://www.kaggle.com/code/altairfarooque/multi-label-image-classification-cv-3-0.

Major modifications include:

* `get_data.py` : This file separates the generated waterfall plots from both Mars and commercial instruments
into the respective `train` and `test` directories.

* `generated_data.py` : This file creates the `training` and `testing` dataloader that can be utilized by the model. 

* `last_layer.py` : In order to generate different probabilities for each of the chemical compounds, modify `MLCNNet`
classify for `n_classes`.

* `CNN_main.py` : This is the main file that runs the model and contains all the model parameters such as
learning rate, number of epochs etc. It also specifies which pretrained resnet network to use for training.
