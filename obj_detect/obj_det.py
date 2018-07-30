import numpy as np
import cv2
import argparse

#constructing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the image")
ap.add_argument("-p","--prototxt",required=True,help="path to caffe 'deploy' prototxt file")
ap.add_argument("-m","--model",required=True,help="path to caffe pre-trained model")
ap.add_argument("-c","--confidence",type=float,default=0.2,help="min probability to help filter out useless predictions")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#loading our model from the disk
print("[info] loading the model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

#loading the input image and constructing an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread(args["image"])
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007843,(300,300),127.5)

#passing the blob to our images so as to get the predictions
print("[info] computing object detections...")
net.setInput(blob)
detections=net.forward()
print("[INFO] detections:",detections)

#loop over the detections
for i in np.arange(0,detections.shape[2]):
    #extract the probability associated with the prediction
    confidence = detections[0,0,i,2]

    #filter out weak detections 
    if confidence > args["confidence"]:
        #extract the index of the class label from the detections,
        #then compute the x,y coordinates for the bounding box for the object
        idx = int(detections[0,0,i,1])
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startx,starty,endx,endy)=box.astype("int")

        #display the predictions
        label = "{}:{:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[info] {}".format(label))
        cv2.rectangle(image,(startx,starty),(endx,endy),COLORS[idx],2)
        y = starty-15 if starty - 15 > 15 else starty + 15
        cv2.putText(image,label,(startx,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)

#show the output image
cv2.imshow("output",image)
cv2.waitKey(0)