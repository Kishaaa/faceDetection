So this code covers projects 4-6

	ORGANIZATION:

FOLDER:
	eyeLike			-	library folder downloaded from Mr. Humes github page
	haar_cascades		-	tar file containing haar classifiers for face, nose etc. from the course website
	res			-	Resource folder which stores the frames of videos with folder-id = video-id
	tmp			-	Temporary folder which stores the processed frames

FILES:
	faceDetection.py	-	code related to haar classification and interface to pupil detection library
	databaseHandler.py	-	functions dealing with the postgre database for this project
	schema.sql		-	database schema used in this project

RUN:
1. compilation:	
	> cd eyeLike
	> cd build
	> cmake ../
	> make
2. run:
	> python faceDetection.py res/sample.mp4
		or
	> python faceDetection.py id <video-id of the existing video>

CONVENTION FOLLOWED:
1. All the output/processed videos are stored in the home folder with the following naming convention:
	bounding_box_eye_pupil_tracking_movie_<video_id>.mp4
2. All the video files are placed in res/ folder
	it is not a requirement

DATABASE:
Type: POSTGRES
	dbname = 'facedetect'
	host = 'localhost'
	port = '5432'
	user = 'postgres'
	password = 'password'

LINKING TO EYELIKE LIBRARY:
main.cpp file in the eyeLike/src folder has been modified to take inputs as:
	<eyelike-executable> frame-image eyebox_x eyebox_y eyebox_width eyebox_height
and it outputs
	pupil_x pupil_y