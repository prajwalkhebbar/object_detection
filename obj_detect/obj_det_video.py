import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import time
import argparse

#constructing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",required=False,help="path to the video")
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

#starting the videostream and fps
print("[info] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2)
fps=FPS().start()

#loop over the frames
while True:
    #grab the frame and resize it to 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame,width=None)

    #grab the frame dimension and convert it to blob
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007843,(300,300),127.5)

    #passing the blob to our images so as to get the predictions
    print("[info] computing object detections...")
    net.setInput(blob)
    detections=net.forward()
    # print("[INFO] detections:",detections)

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
            cv2.rectangle(frame,(startx,starty),(endx,endy),COLORS[idx],2)
            y = starty-15 if starty - 15 > 15 else starty + 15
            cv2.putText(frame,label,(startx,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)

    #show the output frame
    cv2.imshow("output",frame)
    k=cv2.waitKey(1) & 0xFF
    if k == 27:
        print("[info] exiting....")
        break
    
    #update fps counter
    fps.update()

#stop fps counter
fps.stop()

#print the elapsed time and fps
print("[info] elpased time:{:.2f}".format(fps.elapsed()))
print("[info] approx fps:{:.2f}".format(fps.fps()))

#doing some cleanup
cv2.destroyAllWindows()
vs.stop()