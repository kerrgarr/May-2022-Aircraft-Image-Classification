# May-2022-Aircraft-Image_Classification

The FGVC-Aircraft dataset containing 10,000 images of aircraft (covering 100 different models) can be downloaded at 
https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/#:~:text=FGVC%2DAircraft%20Benchmark,375%20KB%20%7C%20MD5%20Sum%5D.

The dataset is discussed in the paper "Fine-Grained Visual Classification of Aircraft" by Maji, et al [2013] which can be found at https://arxiv.org/abs/1306.5151

To start, we want to 
1. visualize the dataset
2. train a deep neural network to classify images of the different aircraft models

Install dependencies using:

<code> python -m pip install -r requirements.txt </code>

and if you are using conda use:

<code> conda env create aircraft.yml </code>

which can be activated for your Python environment using: 

<code> conda activate aircraft </code>

Currently, in the folder code, you can train the models using a multi-layer perceptron model

<code> python -m code.train -m mlp </code>

We will work on CNN and more interesting models, too.

This dataset is tricky; the image sizes are different, so I had to use transform.Resize() on the data. (I forced the images to be 64x64 which may hurt the accuracy of the model. Just as an aside, it would be good to cover data augmentation for image classification -- rotations, grayscale, resizing, etc.) There are also many family classes for the aircraft (70 categories!), so we may consider reducing this dataset to a subset for better handling and visualization.

