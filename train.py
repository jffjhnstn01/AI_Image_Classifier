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

#Argument Parser to collect command line info
ap = argparse.ArgumentParser(description='train.py')
ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learnrate', dest="learnrate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4100)

#Assign parser arguments to function arguments
pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
learnrate = pa.learnrate
structure = pa.arch
dropout = pa.dropout
hiddenlayer1 = pa.hidden_units
device = pa.gpu
epochs = pa.epochs

def main():
    
    #Run load_data function with command line file path
    trainloader, testloader, validloader = utilities.load_data(root)
    #Run network_setup with command line structure, dropout, hiddenlayer number, and learnrate
    model, criterion, optimizer = utilities.network_setup(structure,dropout,hiddenlayer1,learnrate)
    #run deep_learning training function with model, criterion, and optimizer from network_setup and command line arguments
    utilities.deep_learning(model, criterion, optimizer, trainloader, epochs, 40)
    #save the checkpoint of the trained model for later use.
    utilities.save_checkpoint(model,path,structure,hiddenlayer1,dropout,learnrate)
    print("Training complete. Model saved at {}".format(path))


if __name__== "__main__":
    main()        



                    



