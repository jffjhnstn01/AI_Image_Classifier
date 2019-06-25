# AI_Image_Classifier

# Udacity AI Programming with Python Image Classifier Project
This project is used to train an artificial intelligence neural network to identify flower images from a dataset of 102 labelled flowers. The project will produce a command line application that takes an input image file path and identifies the (5) most likely species of flower that the image shows.

# Prerequisites
This project uses Python 3.6.5. You can find the most up to date version of Python at https://www.python.org/  
Additonal packages required are Numpy, Pandas, Matplotlib, Pytorch, PIL and json. Download these packages using pip:  
  `pip install numpy pandas matplotlib pil`  
Pytorch should be installed using documentation on the official website https://pytorch.org/  

# Command Line Application
Train a new network on a data set with train.py  
  Basic usage: `python train.py data_directory`  
    Prints out training loss, validation loss, and validation accuracy as the network trains  
  Options:  
    Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`  
    Choose architecture(vgg16 or densenet121): `python train.py data_dir --arch "vgg13"`  
    Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`  
    Use GPU for training: `python train.py data_dir --gpu`  
    
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.  

  Basic usage: `python predict.py /path/to/image checkpoint`  
  Options:  
    Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`  
    Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`  
    Use GPU for inference: `python predict.py input checkpoint --gpu`  
    
# Data
The data used for this project are a flower database(.json file).  
The data need to comprised of 3 folders:  
-test  
-train  
-validate  
Generally the proportions should be 70% training 10% validate and 20% test.

Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file. For example if we have the image x.jpg and it is a tulip it could be in a path like this /test/5/x.jpg and json file would be like this {...5:"tulip",...}

# GPU
