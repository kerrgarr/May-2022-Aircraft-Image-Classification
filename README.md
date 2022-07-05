# May-2022-Aircraft-Image_Classification

The FGVC-Aircraft dataset containing 10,000 images of aircraft (covering 100 different models) can be downloaded at 
https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/#:~:text=FGVC%2DAircraft%20Benchmark,375%20KB%20%7C%20MD5%20Sum%5D.

The dataset is discussed in the paper "Fine-Grained Visual Classification of Aircraft" by Maji, et al [2013] which can be found at https://arxiv.org/abs/1306.5151

<!-- 
I modified the file structure a bit and have the re-structured code available for download at:

https://www.dropbox.com/s/3ph6z4n5qwmz6v5/data.zip?dl=0

(Please let me know if the link does not work.) 
-->


This dataset is tricky; the image sizes are different, so I had to use transform.Resize() on the data. (I forced the images to be 64x64 which may hurt the accuracy of the model. Just as an aside, it would be good to cover data augmentation for image classification -- rotations, grayscale, resizing, etc.) 

There are also many family classes for the aircraft (70 categories!), so I chose to reduce it to just the Airbus fleets (6 classes): ['A300','A310','A320','A330','A340','A380'].

We want to 
1. visualize the dataset
2. train a deep neural network to classify images of the different aircraft models (Linear, MLP, CNN, ResNet)

For logging later on, optionally add a folder called log_test. Your directory structure should look like:
```
May-2022-Aircraft-Image-Classification

log_test/
data/
   train
      (various image files...)
       labels.csv
   valid
      (various image files...)
       labels.csv
code
   __init__.py
   logging.py
   models.py
   train.py
   train_cnn.py
   utils.py
   PlotCNNprediction.py
   PlotDatasetImages.py
```
 ---------------------------------------------------------------------------------------
 
Install dependencies using:

<code> python -m pip install -r requirements.txt </code>

and if you are using conda use:

<code> conda env create aircraft.yml </code>

which can be activated for your Python environment using: 

<code> conda activate aircraft </code>

--------------------------------------------------------------------------------------------------------
Currently, in the folder code, you can train the models using a linear model, multi-layer perceptron model, convolutional neural network, and ResNet 152.

To train the Linear, the code is: 

<code> python -m code.train -m linear </code>

To train the MLP, the code is: 

<code> python -m code.train -m mlp </code>

To train the CNN, the code is: 

<code> python -m code.train -m cnn </code>

To train the ResNet, the code is: 

<code> python -m code.train -m resnet </code>

---------------------------------------------------------------------

To do some visualization:

Here is the code you can run before training the model to see a snapshot of what the dataset looks like:

<code> python -m code.PlotDatasetImages data/train  </code>

If you want to train with the CNNClassifier model in models.py, run the code:

<code> python -m code.train -m cnn -n 1000 </code> 

This runs the CNN classifier for 1000 epochs in train.py

ResNet152 is a prebuilt image classification network that should beat our home-built CNN classifier. To run and test it, run

<code> python -m code.train -m resnet -n 1000 </code> 

To visualize the results of our model on the validation data, you can plot the following:

<code> python -m code.PlotCNNprediction -model resnet -dataset data/valid </code>

What I see is that it performs with accuracy ~40 %, even with ResNet on the validation data. It get ~100% accuracy on training set:

<code> python -m code.PlotCNNprediction -model resnet -dataset data/train </code>

This means ResNet and CNN are overfitting to our training data ! (Not good.)

_____________________________________________________________________
For Windows, if you want to use Tensorboard, here is some extra code:

<code> python -m code.train_cnn --log_dir log_test -n 1000 </code>

<code> python -m code.logging log_test </code>

<code> tensorboard --logdir=log_test --port 6006 --bind_all  </code>
             
the message you'll receive will give you something like:

http://something-like:6006/

click on the address you get and open it in a web browser. See the interactive tensorboard. Done!


