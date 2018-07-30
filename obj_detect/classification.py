import numpy as np
import argparse
import time
import cv2
import sys
#construction the parameters
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
ap.add_argument("-p","--prototxt",required=True,help="path to caffe 'deploy' prototxt file")
ap.add_argument("-m","--model",required=True,help="path to caffe pre-trained model")
ap.add_argument("-l","--labels",required=True,help="path to ImageNet labels(i.e,syn-sets)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
#loading the labels from the disks
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# our CNN requires fixed spatial dimensions for our input image(s)
# so we need to ensure it is resized to 224x224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

#loading our seerialized model from disks
print("[info] loading model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

#set the blob as input to the network and perform a forward-pass to
#obtain our output classification
net.setInput(blob)
start = time.time()
predictions = net.forward()
end = time.time()
print("[info] classification took around {:.5}".format(end-start))

#sorting the indexes of the probabilities in descending order and grabing
#the top 5 predictions
idxs=np.argsort(predictions[0])[::-1][:5]

#loop over the top 5 predictions and displaying them
for(i, idx) in enumerate(idxs):
    #drawing he top prediction on the input image
    if i==0:
        text = "label: {}. {:.2f}%".format(classes[idx],predictions[0][idx]*100)
        cv2.putText(image,text,(5,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        #display the output image
        cv2.imshow("image",image)
        k=cv2.waitKey(0)


    #display the predicted label + associated probability to the console
    print("[info] {}, label:{}, probability:{:.5}".format(i+1,classes[idx],predictions[0][idx]))
