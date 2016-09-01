import psycopg2, sys

class DBHandler:
	def __init__(self):
		self.dbname = 'facedetect'#'cs161'
		self.host = 'localhost'
		self.port = '5432'
		self.user = 'postgres'
		self.password = 'password'
		self.con = psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, password=self.password)

	def connectDB(self):
		self.con = psycopg2.connect(dbname=self.dbname, user=self.user, host=self.host, password=self.password)

	def createDB(self):			
		pass

	def insert_videometa(self, vname, num_frames, x_res, y_res, fps):	
		try:
			if self.con is None:
				self.connectDB()			
			cursor = self.con.cursor()

			#check if the video name already exists
			try:
				query = "select video_id from video_meta where name ='"+vname+"';"
				cursor.execute(query)
				rows = cursor.fetchall()
				if rows is not None and len(rows) > 0:
					print 'DBHandler: Video file already exists in the database, video_id:', rows[0][0]
					return int(rows[0][0]) #returning video id
			except:
				print 'no video with such name exits'

			query =  "INSERT INTO video_meta (name, num_frames, x_resolution, y_resolution, fps) VALUES (%s, %s, %s, %s, %s) returning video_id;"
			vals = (vname, num_frames, x_res, y_res, fps)
			cursor.execute(query, vals)
			self.con.commit()
			return int(cursor.fetchone()[0]) #video id of the new video
		except:
			print('DBHandler: inserting video metadata: unexpected error occured', sys.exc_info())

	def insert_boundingboxdata(self, v_id, frame_num, boxtype, x, y, w, h):
		try:
			if self.con is None:
				self.connectDB()			
			cursor = self.con.cursor()

			#check if the video name already exists
			try:
				query = "select bbox_id from bounding_boxes where video_id ='"+str(v_id)+"' and frame_number='"+str(frame_num)+"' and bbox_type='"+boxtype+"';"
				cursor.execute(query)
				rows = cursor.fetchall()
				if rows is not None and len(rows) > 0:
					print 'DBHandler: bbox already exists in the database, bbox_id:', rows[0][0]
					return int(rows[0][0]) #returning video id
			except:
				print 'no existing bbox for the same v_id, frame_num, and type',

			query =  "INSERT INTO bounding_boxes (video_id, frame_number, bbox_type, upper_left_corner_coord, width, height) VALUES (%s, %s, %s, point(%s, %s), %s, %s) returning bbox_id;"
			vals = (v_id, frame_num, boxtype, x, y, w, h)
			cursor.execute(query, vals)
			self.con.commit()
			return int(cursor.fetchone()[0]) #bbox id of the new video
		except:
			print('DBHandler: inserting bbox metadata: unexpected error occured', sys.exc_info())


	def insert_pupildata(self, v_id, frame_num, lpupilx, lpupily, rpupilx, rpupily):
		try:
			if self.con is None:
				self.connectDB()			
			cursor = self.con.cursor()

			#check if the video name already exists
			try:
				query = "select pupils_id from pupils where video_id ='"+str(v_id)+"' and frame_number='"+str(frame_num)+"';"
				cursor.execute(query)
				rows = cursor.fetchall()
				if rows is not None and len(rows) > 0:
					print 'DBHandler: pupil already exists in the database, bbox_id:', rows[0][0]
					return int(rows[0][0]) #returning video id
			except:
				print 'no existing pupil for the same v_id, frame_num, and type',

			query =  "INSERT INTO pupils (video_id, frame_number, left_eye_coords, right_eye_coords) VALUES (%s, %s, point(%s, %s), point(%s, %s)) returning pupils_id;"
			vals = (v_id, frame_num, lpupilx, lpupily, rpupilx, rpupily)
			cursor.execute(query, vals)
			self.con.commit()
			return int(cursor.fetchone()[0]) #pupils_id of the new video
		except:
			print('DBHandler: inserting pupil metadata: unexpected error occured', sys.exc_info())

	def getDB_frame(self, frameobj):
		pass
