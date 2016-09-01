import cv2, sys, time, subprocess, os
from databaseHandler import DBHandler
from ffprobe import FFProbe

class faceDetect:
    def __init__(self, _num_frames=0, _frame_rate=0, _width=0, _height=0, _vid_id=0, _vid_name='', _dbhandle=None):
        # input#setting up the cascase classifiers for different sections of the face
        self.frontalFaceCascade = cv2.CascadeClassifier("haar_cascades/haarcascade_frontalface_alt2.xml")
        self.frontalEyeCascade = cv2.CascadeClassifier("haar_cascades/frontaleye.xml")
        self.leftEyeCascade = cv2.CascadeClassifier("haar_cascades/lefteye.xml")
        self.rightEyeCascade = cv2.CascadeClassifier("haar_cascades/righteye.xml")
        self.noseCascade = cv2.CascadeClassifier("haar_cascades/nose.xml")
        self.mouthCascade = cv2.CascadeClassifier("haar_cascades/mouth.xml")

        self.num_frames = _num_frames
        self.frame_number = 1
        self.frame_rate = _frame_rate
        self.width = _width
        self.height = _height
        self.video_id = _vid_id
        self.video_name = _vid_name

        self.vframes = []
        self.metainfo = {}

        self.dbhandle = _dbhandle

    def draw_crosshairs(self, img, point, _len, _color):
        cv2.line(img, (point[0]-_len, point[1]), (point[0]+_len, point[1]), _color, 3);  #crosshair horizontal
        cv2.line(img, (point[0], point[1]-_len), (point[0], point[1]+_len), _color, 3);  #crosshair vertical

    def queryVideoMetadata(self, _vid_name):
        try:
            metadata = FFProbe(_vid_name)
            if len(metadata.streams) == 0: 
                _num_frames = -1
                return None

            for stream in metadata.streams:
                if stream.isVideo():
                    self.num_frames = stream.frames()
                    self.frame_rate = stream.r_frame_rate.split('/')[0]
                    self.width = stream.width
                    self.height = stream.height
                    self.frame_number = 1
                    return [self.num_frames, self.width, 
                            self.height, self.frame_rate]
        except:
            _num_frames = -1
            return None

    def extractFrames(self, _vid_name, count = 0):
        vidcap = cv2.VideoCapture(_vid_name)        
        success,image = vidcap.read()
        while success:            
            success,image = vidcap.read()
            count += 1
            yield (count,image)

    def detectPupil(self, frame, ex, ey, ew, eh): #detect one pupil at a time - computationally fast
        try:
            cv2.imwrite("res/frame.png", frame)
            cmd = ["./eyeLike/build/bin/eyeLike", "res/frame.png", str(ex), str(ey), str(ew), str(eh)]
            result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)            
            out,err = result.communicate()
            return out.strip()
        except:
            print('eyeLike library error: ', sys.exc_info())
            return

    def haarClassification(self, frame, pupilDetection=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.frontalFaceCascade.detectMultiScale(
            gray,
            scaleFactor=1.01,
            minNeighbors=10,
            minSize=(30, 30),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
        )

        # If a face is detected
        for i,(x, y, w, h) in zip([0], faces): #only for the first face
            # --------------------------- EYE detection --------------------------        
            # here we are selecting the region of interest
            # Got the exact values of boxes from the example on course website
            face_color = frame[y:y+h, x:x+w]
            lefteye_gray = gray[y+int(h/5):y+int(h/5)+int(h/3), x+int(w/2):x+w]
            righteye_gray = gray[y+int(h/5):y+int(h/5)+int(h/3), x:x+int(w/2)]

            lefteye_color = frame[y+int(h/5):y+int(h/5)+int(h/3), x+int(w/2):x+w]
            righteye_color = frame[y+int(h/5):y+int(h/5)+int(h/3), x:x+int(w/2)]

            # setting minimum and maximum boundaries for the eyes
            minimum_width_left_eye =  int(0.18 * w);
            minimum_height_left_eye = int(0.14 * h);
            if (minimum_width_left_eye < 18) or (minimum_height_left_eye < 12) :
                minimum_width_left_eye   = 18;
                minimum_height_left_eye  = 12;
            maximum_width_left_eye   = minimum_width_left_eye * 2;
            maximum_height_left_eye  = minimum_height_left_eye * 2;

            minimum_width_right_eye =  minimum_width_left_eye;
            minimum_height_right_eye = minimum_height_left_eye;
            maximum_width_right_eye  = maximum_width_left_eye;
            maximum_height_right_eye = maximum_height_left_eye;

            # applying the classifiers
            leftEye = self.leftEyeCascade.detectMultiScale(
                lefteye_gray,
                scaleFactor=1.01,
                minNeighbors=10,
                minSize=(minimum_width_left_eye, minimum_height_left_eye),
                maxSize=(maximum_width_left_eye, maximum_height_left_eye),
                flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
            )

            rightEye = self.rightEyeCascade.detectMultiScale(
                righteye_gray,
                scaleFactor=1.01,
                minNeighbors=10,
                minSize=(minimum_width_right_eye, minimum_height_right_eye),
                maxSize=(maximum_width_right_eye, maximum_height_right_eye),
                flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
            )

            # -------------------------- NOSE detection --------------------------
            minimum_nose_width = 25
            minimum_nose_height = 15
            nose_gray = gray[y+int(h/5)+int(h/3):y+int((3*h)/4), x:x+w]
            nose_color = frame[y+int(h/5)+int(h/3):y+int((3*h)/4), x:x+w]

            nose = self.noseCascade.detectMultiScale(
                nose_gray,
                scaleFactor=1.01,
                minNeighbors=10,
                minSize=(minimum_nose_width, minimum_nose_height),
                flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
            )

            # ------------------------- MOUTH detection --------------------------
            minimum_mouth_width = 25
            minimum_mouth_height = 15
            mouth_gray = gray[y+int((3*h)/4):y+h, x:x+w]
            mouth_color = frame[y+int((3*h)/4):y+h, x:x+w]

            mouth = self.mouthCascade.detectMultiScale(
                mouth_gray,
                scaleFactor=1.01,
                minNeighbors=10,
                minSize=(minimum_mouth_width, minimum_mouth_height),
                flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
            )

            # ------------------------ DRAWING section ------------------------------------
            # This section also does pupil detection and bounding box insertion into db

            #face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #inserting bounding box in db - face
            try:
                self.dbhandle.insert_boundingboxdata(self.video_id, self.frame_number, 'Face', int(x), int(y), int(w), int(h))
            except:
                print 'MAIN: face bounding box error'

            #left eye
            xleftpupil = yleftpupil = xrightpupil = yrightpupil = 0
            for i,(ex,ey,ew,eh) in zip([0], leftEye): #accessing only the first element                
                if pupilDetection:
                    [xleftpupil, yleftpupil] = [int(x) for x in self.detectPupil(lefteye_color,
                                                                                ex,
                                                                                ey,
                                                                                ew,
                                                                                eh).strip().split(' ')]
                    self.draw_crosshairs(lefteye_color, (xleftpupil, yleftpupil), 20, (0,255,255))
                    self.metainfo['leftpupil'] = [xleftpupil, yleftpupil]

                cv2.rectangle(lefteye_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                self.metainfo['lefteye'] = [ex, ey, ew, eh]
                # inserting bounding box in db - left eye
                try:
                    self.dbhandle.insert_boundingboxdata(self.video_id, self.frame_number, 'LEye', int(ex), int(ey), int(ew), int(eh))
                except:
                    print 'MAIN: left eye bounding box error'

            #right eye
            for i,(ex,ey,ew,eh) in zip([0], rightEye): #accessing only the first element
                if pupilDetection:
                    [xrightpupil, yrightpupil] = [int(x) for x in self.detectPupil(righteye_color,
                                                                                ex,
                                                                                ey,
                                                                                ew,
                                                                                eh).strip().split(' ')]
                    self.draw_crosshairs(righteye_color, (xrightpupil, yrightpupil), 20, (0,255,255))
                    self.metainfo['rightpupil'] = [xrightpupil, yrightpupil]
                cv2.rectangle(righteye_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
                self.metainfo['righteye'] = [ex, ey, ew, eh]
                # inserting bounding box in db - right eye
                try:
                    self.dbhandle.insert_boundingboxdata(self.video_id, self.frame_number, 'REye', int(ex), int(ey), int(ew), int(eh))
                except:
                    print 'MAIN: right eye bounding box error'

            #insert pupil data into db
            self.dbhandle.insert_pupildata(self.video_id, self.frame_number, int(xleftpupil), int(yleftpupil), int(xrightpupil), int(yrightpupil))

            #nose
            for i,(ex,ey,ew,eh) in zip([0], nose): #accessing only the first element
                cv2.rectangle(nose_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
                self.metainfo['nose'] = [ex, ey, ew, eh]
                # inserting bounding box in db - nose
                try:
                    self.dbhandle.insert_boundingboxdata(self.video_id, self.frame_number, 'Nose', int(ex), int(ey), int(ew), int(eh))
                except:
                    print 'MAIN: nose bounding box error'

            #mouth
            for i,(ex,ey,ew,eh) in zip([0], mouth): #accessing only the first element
                cv2.rectangle(mouth_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
                self.metainfo['mouth'] = [ex, ey, ew, eh]
                # inserting bounding box in db - mouth
                try:
                    self.dbhandle.insert_boundingboxdata(self.video_id, self.frame_number, 'Mouth', int(ex), int(ey), int(ew), int(eh))
                except:
                    print 'MAIN: mouth bounding box error'

        return frame
        #cv2.imshow('Video', gray)


    def processFrames(self, webcam=False, createTemp=False, tempAddr=''):
        # input
        count = 1
        if webcam:
            video_capture = cv2.imread("res/"+self.video_id+"")

        #fourcc = cv2.VideoWriter_fourcc('H','2','6','4') #video codec - 4 character code        
        while True:
        #for i in range(1):
            # Capture frame-by-frame
            try:
                print 'Processing frame-id: %d'%count
                if webcam: 
                    ret, orig_frame = video_capture.read()
                else: 
                    orig_frame = cv2.imread("res/"+self.video_id+"/frame%d.jpg"%count)
                    if orig_frame is None:
                        break
                frame = orig_frame.copy()
                # ------------- face, eyes, nose, mouth detection --------------------
                newframes = self.haarClassification(frame, pupilDetection=True)

                self.vframes.append(newframes)

                if createTemp: # dump processed images in the temp directory
                    cv2.imwrite(tempAddr+"frame%d.jpg"%count, newframes)

                if webcam:  #if using webcam show the feed on a seperate window
                    cv2.imshow("Video",frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                count += 1
                self.frame_number = count
            except:
                print('MAIN: unexpected error occured', sys.exc_info())
                if webcam:
                    video_capture.release()
                break

        #cv2.waitKey(0)
        # When everything is done, release the capture
        if webcam: 
            video_capture.release()
            video_capture = cv2.VideoCapture(0)
        cv2.destroyAllWindows()

    def saveFrames(self):
        name = 'bounding_box_'+'eye_pupil_tracking_movie_'+self.video_id+'.mp4'
        print 'saving the video=', name, ' width=',self.width,' height=',self.height,' fps=',self.frame_rate,' Number of frames=',len(self.vframes)
        if int(self.frame_rate) > 30:
            self.frame_rate = 30
        video_out = cv2.VideoWriter(name,cv2.VideoWriter_fourcc('x','2','6','4'), 
                                    int(self.frame_rate), (int(self.width), int(self.height)))

        for i,frames in enumerate(self.vframes):
            video_out.write(frames)
        video_out.release()


if __name__ == '__main__':    
    input_from_webcam = False #False: use user-defined video, True: use webcam
    dbhandler = DBHandler()
    task = faceDetect(_dbhandle=dbhandler)

    if not input_from_webcam:
        # take input from cmd line    
        if len(sys.argv) < 2: 
            print 'Usage: python faceDetection.py <video-file-url>'
            print 'Usage: python faceDetection.py id <video-id>'
            exit(1)

        if len(sys.argv) == 3: #passing video id
            if sys.argv[1] == 'id':                
                print 'running algorithm on video with id = ',sys.argv[2]
                video_id = sys.argv[2]
                task.video_id = video_id
            else:
                print 'Usage: python faceDetection.py id <video-id>'
                print 'Usage: python faceDetection.py <video-file-url>'
                exit(1)

            if not os.path.exists("res/"+video_id):
                print "Video folder NOT found or empty on the system"
                exit(1)
            else:
                print "MAIN: Video folder found on the system"

        else:
            video_file = sys.argv[1].strip()
            #if file name doesn't exist in the database then insert it in the database
            metainfo = task.queryVideoMetadata(video_file)
            print 'Metainformation about the video passed: ', metainfo
            if metainfo is not None:
                #valid video file
                print 'MAIN: reading video metainformation and pushing it to db'
                video_id = dbhandler.insert_videometa(video_file.split('/')[-1], metainfo[0], 
                    metainfo[1], metainfo[2], metainfo[3])            
                video_id = str(video_id)
                task.video_id = video_id
                if not os.path.exists("res/"+video_id):
                    print('MAIN: Extracting frames at '+"res/"+video_id+"/")
                    os.makedirs("res/"+video_id)
                    for (count,frame) in task.extractFrames(video_file):
                        if frame is not None:
                            #found valid frame - save it in folder
                            cv2.imwrite("res/"+video_id+"/frame%d.jpg" % count, frame)
                else:
                    print "MAIN: Video folder already exists on the system"
            else:
                print 'Video file NOT found'
                exit(1)

    #else query database for the frames
    print 'MAIN: Processing Frames of video with id:', task.video_id
    task.processFrames(webcam=input_from_webcam, createTemp=True, tempAddr='tmp/')
    print 'MAIN: Output the video'
    task.saveFrames()