# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)
# import the necessary packages
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import matplotlib.pyplot as plt
import argparse
import imutils
import torch
import cv2

def test(model, testDataLoader, testData, device, H):
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []
        # loop over the test set
        for (x, y) in testDataLoader:
            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
            
    # generate a classification report
    #print(classification_report(testData.targets.cpu().numpy(), np.array(preds), target_names=testData.classes))
    print(classification_report(np.array([y for _, y in testData]), np.array(preds), target_names=testData.dataset.classes))

    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")
    # serialize the model to disk
    torch.save(model, "model.pth")
    
    # set the device we will be using to test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the KMNIST dataset and randomly grab 5 data points
    print("[INFO] loading the test dataset...")
    idxs = np.random.choice(range(0, len(testData)), size=(5,))
    testData = Subset(testData, idxs)
    # initialize the test data loader
    testDataLoader = DataLoader(testData, batch_size=1)
    # load the model and set it to evaluation mode
    model = torch.load("model.pth").to(device)
    model.eval()
    
    # switch off autograd
    with torch.no_grad():
        # loop over the test set
        for (image, label) in testDataLoader:
            # grab the original image and ground truth label
            origImage = image.numpy().squeeze()
            gtLabel = testData.dataset.classes[label.numpy()[0]]
            # send the input to the device and make predictions on it
            image = image.to(device)
            pred = model(image)
            # find the class label index with the largest corresponding
            # probability
            idx = pred.argmax(axis=1).cpu().numpy()[0]
            predLabel = testData.dataset.classes[idx]
        
            # convert the image from grayscale to RGB (so we can draw on
            # it) and resize it (so we can more easily see it on our
            # screen)
            origImage = np.dstack([origImage] * 3)
            origImage = imutils.resize(origImage, width=128)
            # draw the predicted class label on it
            color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
            cv2.putText(origImage, gtLabel, (2, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
            # display the result in terminal and show the input image
            print("[INFO] ground truth label: {}, predicted label: {}".format(
                gtLabel, predLabel))
            #cv2.imshow("image", origImage)
            #cv2.waitKey(0)