#Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable
import json
import PIL
from PIL import Image
import argparse
import utilities

#Argument parser for command line functions
ap = argparse.ArgumentParser(description='predict.py')

ap.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
ap.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
ap.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

#Parameters from argument parser
pa = ap.parse_args()
path_image = pa.input
number_of_outputs = pa.top_k
device = pa.gpu
path = pa.checkpoint
pa = ap.parse_args()

def main():
    #Load checkpoint from saved path if available
    model=utilities.load_checkpoint(path)
    #get categories from json file
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
    #Predict picture category using trained model
    probabilities = utilities.predict(path_image, model, number_of_outputs)
    #Get labels and probabilities of prediciton for printing
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    i=0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1

    
if __name__== "__main__":
    main()