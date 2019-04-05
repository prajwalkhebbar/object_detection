import sys
import cv2

(major_ver,minor_ver,sub_ver)=(cv2.__version__).split(".")

if __name__ == '__main__':

    #set up tracker
    tracker_types=['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']
    tracker_type=tracker_types[1]

    # if int(minor_ver) < 3:
    #     tracker = cv2.Tracker_create(tracker_type)
    # else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    #read video
    video = cv2.VideoCapture(0)

    #exit if video not opened
    if not video.isOpened():
        print ("could not open the video")
        sys.exit()
    #read the first frame
    ok, frame = video.read()
    if not ok:
        print ("cannot read the video frame")
        sys.exit()

    #defining the initial bounding box
    bbox = cv2.selectROI(frame,False)

    #initialize tracker with first frame and bounding box
    ok = tracker.init(frame,bbox)

    while True:
        #read a new frame
        ok,frame = video.read()
        if not ok:
            break
        #start timer
        timer = cv2.getTickCount()

        #update tracker
        ok, bbox = tracker.update(frame)

        #calculate fps
        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)

        #draw the bounding box
        if ok:
            p1=(int(bbox[0]),int(bbox[1]))
            p2=(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
            cv2.rectangle(frame,p1,p2,(233,234,23),2,1)
        else:
            cv2.putText(frame,"tracking failed",(100,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,233),2)

        #display tracker type
        cv2.putText(frame,tracker_type+" tracker",(100,20),cv2.FONT_HERSHEY_SIMPLEX,.75,(233,42,52),2)

        #display fps
        cv2.putText(frame,str(int(fps))+" fps",(100,50),cv2.FONT_HERSHEY_SIMPLEX,.75,(33,242,52),2)

        #display result
        cv2.imshow("tracking",frame)

        #exit if esc is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27 :break
        