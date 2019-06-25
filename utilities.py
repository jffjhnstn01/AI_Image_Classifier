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

#Dictionary of pretrained networks and the number of inputs
networks = {'vgg16':25088,
        'densenet121':1024}

#Transform images in training, validation, and testing sets
def image_transform(path_root):
    data_dir = path_root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #Training set gets randomized rotations, cropping, and horizontal flips for robust training
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    #Resize and centercrop test and validation sets, no randomization
    test_transforms = transforms.Compose([transforms.Resize(256), 
                                          transforms.CenterCrop(224), 
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  
    
    valid_transforms = transforms.Compose([transforms.Resize(256), 
                                           transforms.CenterCrop(224), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #Combine training, testing, and validation data to be used later
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms) 
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms) 
    valid_data= datasets.ImageFolder(valid_dir, transform=valid_transforms)

    return train_data, test_data, valid_data

def load_data(path_root):
    data_dir = path_root
    train_data, test_data, valid_data = image_transform(path_root)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True) 
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32) 
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return trainloader, testloader, validloader

#Run the image transform function to get training, validation, and testing data
train_data,test_data,valid_data=image_transform('./flowers/')
trainloader,testloader,validloader=load_data('./flowers/')

def network_setup(structure='vgg16',dropout=0.2, hiddenlayer1 = 4100,learnrate = 0.001):
    #Select pretrained network from dictionary based on function argument
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Please enter either vgg16 or densenet121")
    #Set device to GPU if available  
    device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(nn.Linear(networks[structure],hiddenlayer1), 
                               nn.ReLU(), 
                               nn.Dropout(dropout), 
                               nn.Linear(hiddenlayer1, 1000), 
                               nn.ReLU(), 
                               nn.Dropout(dropout), 
                               nn.Linear(1000, 102), 
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learnrate )
    return model, criterion, optimizer

def deep_learning(model, criterion, optimizer, trainloader, epochs=5, print_every=40):
    steps = 0
    #choose the correct device based on what's available
    device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    #set device to cuda if available
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            #Forward pass through neural network
            logps = model.forward(inputs)
            #Calculate loss of output values
            loss = criterion(logps, labels)
            #Backpropagation through neural network to update weights
            loss.backward()
            #update parameters based on new gradient
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                validation_accuracy = 0

                for ii, (inputs_v, labels_v) in enumerate(validloader):
                    #Intialize gradients to zero
                    optimizer.zero_grad()
                    inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)
                    model.to(device)
                    with torch.no_grad():
                        #forward pass with validation set
                        logps = model.forward(inputs_v)
                        #calculate loss
                        validation_loss = criterion(logps, labels_v)
                        #take exponential of outputs
                        ps = torch.exp(logps).data
                        #Determine if output is equivalent to label
                        equals = (labels_v.data == ps.max(1)[1])
                        #Take sum of true equals tensors
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor))

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(validation_accuracy/len(validloader))) 
                running_loss = 0
                model.train()
    return

         
def save_checkpoint(model=0, path='checkpoint.pth', structure='vgg16', hiddenlayer1=4100, dropout=0.2, learnrate=0.001, epochs=5):
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    checkpoint = {'structure': structure,
                  'hiddenlayer1': hiddenlayer1,
                  'dropout': dropout,
                  'learnrate': learnrate,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, path)
    return

def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hiddenlayer1']
    model,_,_ = network_setup(structure, 0.2, hidden_layer1, learnrate=0.001)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    proc_img = Image.open(image)
    preprocess_img = transforms.Compose([transforms.Resize(256), 
                     transforms.CenterCrop(224), 
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    model_img = preprocess_img(proc_img)
    return model_img 
                    
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    model.to(device)
    processed_image = process_image(image_path)
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    
    with torch.no_grad():
           output = model.forward(processed_image.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)
