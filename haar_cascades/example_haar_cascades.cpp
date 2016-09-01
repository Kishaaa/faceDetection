#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main (int argc, char *argv[]) 
{
  Mat source_image_gray, source_image_gray_face_region, source_image_gray_left_eye_region, source_image_gray_right_eye_region;
  CascadeClassifier face_cascade, left_eye_cascade, right_eye_cascade;
  std::vector<Rect> face, eyes;
  char input_filename[1280];
  int number_of_left_eyes_detected, number_of_right_eyes_detected, min_neighbors;
  int maximum_width_left_eye, maximum_height_left_eye, maximum_width_right_eye, maximum_height_right_eye;
  int minimum_width_left_eye, minimum_height_left_eye, minimum_width_right_eye, minimum_height_right_eye;
  int bounding_box_face_x, bounding_box_face_y, bounding_box_face_width, bounding_box_face_height;
  int bounding_box_left_eye_x, bounding_box_left_eye_y, bounding_box_left_eye_width, bounding_box_left_eye_height;
  int bounding_box_right_eye_x, bounding_box_right_eye_y, bounding_box_right_eye_width, bounding_box_right_eye_height;
  float minimum_width_bounding_box, minimum_height_bounding_box, maximum_width_bounding_box, maximum_height_bounding_box;
  double scale_factor;

  min_neighbors = 10;
  scale_factor  = 1.01;
  strcpy (&input_filename[0], "test_image.png");
//
// Load additional cascades here for nose (Nariz.xml) and mouth (Mouth.xml).
// For Nariz.xml, minimum width = 25, minimum height = 15
// For Mouth.xml, minimum width = 25, minimum height = 15
//
  if (face_cascade.load ("./haarcascade_frontalface_alt2.xml") && left_eye_cascade.load ("./lefteye.xml") && right_eye_cascade.load ("./righteye.xml"))
  {
    bounding_box_face_x           = 0;
    bounding_box_face_y           = 0;
    bounding_box_face_width       = 0;
    bounding_box_face_height      = 0;
    bounding_box_left_eye_x       = 0;
    bounding_box_left_eye_y       = 0;
    bounding_box_left_eye_width   = 0;
    bounding_box_left_eye_height  = 0;
    bounding_box_right_eye_x      = 0;
    bounding_box_right_eye_y      = 0;
    bounding_box_right_eye_width  = 0;
    bounding_box_right_eye_height = 0;
    source_image_gray = imread (input_filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (!source_image_gray.empty())
    {
      equalizeHist (source_image_gray, source_image_gray);
      face_cascade.detectMultiScale (source_image_gray, face, scale_factor, min_neighbors, CV_HAAR_FIND_BIGGEST_OBJECT);
      if (face.size() > 0)
      {
        bounding_box_face_x      = face[0].x;
        bounding_box_face_y      = face[0].y;
        bounding_box_face_width  = face[0].width;
        bounding_box_face_height = face[0].height;
        minimum_width_left_eye =  0.18 * bounding_box_face_width;
        minimum_height_left_eye = 0.14 * bounding_box_face_height;
        minimum_width_right_eye =  minimum_width_left_eye;
        minimum_height_right_eye = minimum_height_left_eye;
        if ((minimum_width_left_eye < 18) || (minimum_width_right_eye < 18) || (minimum_height_left_eye < 12) || (minimum_height_right_eye < 12))
        {
          minimum_width_left_eye   = 18;
          minimum_height_left_eye  = 12;
          minimum_width_right_eye  = 18;
          minimum_height_right_eye = 12;
        }
        maximum_width_left_eye   = minimum_width_left_eye * 2;
        maximum_height_left_eye  = minimum_height_left_eye * 2;
        maximum_width_right_eye  = minimum_width_right_eye * 2;
        maximum_height_right_eye = minimum_height_right_eye * 2;
        source_image_gray_left_eye_region = source_image_gray (Rect (bounding_box_face_x + bounding_box_face_width / 2, bounding_box_face_y + bounding_box_face_height / 5.5, bounding_box_face_width / 2, bounding_box_face_height / 3)); 
        left_eye_cascade.detectMultiScale (source_image_gray_left_eye_region, eyes, scale_factor, min_neighbors, CV_HAAR_SCALE_IMAGE, Size(minimum_width_left_eye, minimum_height_left_eye), Size(maximum_width_left_eye, maximum_height_left_eye));
        number_of_left_eyes_detected = eyes.size();
        if (eyes.size() > 0)
        {
          bounding_box_left_eye_x      = bounding_box_face_x + bounding_box_face_width / 2 + eyes[0].x;
          bounding_box_left_eye_y      = bounding_box_face_y + bounding_box_face_height / 5.5 + eyes[0].y;
          bounding_box_left_eye_width  = eyes[0].width;
          bounding_box_left_eye_height = eyes[0].height;
        }
        source_image_gray_right_eye_region = source_image_gray (Rect (bounding_box_face_x, bounding_box_face_y + bounding_box_face_height / 5.5, bounding_box_face_width / 2, bounding_box_face_height / 3)); 
        number_of_right_eyes_detected = 0;
        right_eye_cascade.detectMultiScale (source_image_gray_right_eye_region, eyes, scale_factor, min_neighbors, CV_HAAR_SCALE_IMAGE, Size(minimum_width_right_eye, minimum_height_right_eye), Size(maximum_width_right_eye, maximum_height_right_eye));
        number_of_right_eyes_detected = eyes.size();  
        if (eyes.size() > 0)
        {
          bounding_box_right_eye_x      = bounding_box_face_x + eyes[0].x;
          bounding_box_right_eye_y      = bounding_box_face_y + bounding_box_face_height / 5.5 + eyes[0].y;
          bounding_box_right_eye_width  = eyes[0].width;
          bounding_box_right_eye_height = eyes[0].height;
        }






      }
    }
  }



//  if ((bounding_box_face_x > 0) && (bounding_box_face_y > 0))
//  {
//    rectangle (source_image_gray, Point(bounding_box_face_x, bounding_box_face_y), Point(bounding_box_face_x + bounding_box_face_width, bounding_box_face_y + bounding_box_face_height), Scalar (255,255,255), 1, 8, 0); 
//    imshow ("Image", source_image_gray);
//    waitKey (0);
//  }
//  if ((bounding_box_left_eye_x > 0) && (bounding_box_left_eye_y > 0))
//  {
//    rectangle (source_image_gray, Point(bounding_box_left_eye_x, bounding_box_left_eye_y), Point(bounding_box_left_eye_x + bounding_box_left_eye_width, bounding_box_left_eye_y + bounding_box_left_eye_height), Scalar (255,255,255), 1, 8, 0); 
//    imshow ("Image", source_image_gray);
//    waitKey (0);
//  }
  if ((bounding_box_right_eye_x > 0) && (bounding_box_right_eye_y > 0))
  {
    rectangle (source_image_gray, Point(bounding_box_right_eye_x, bounding_box_right_eye_y), Point(bounding_box_right_eye_x + bounding_box_right_eye_width, bounding_box_right_eye_y + bounding_box_left_eye_height), Scalar (255,255,255), 1, 8, 0); 
    imshow ("Image", source_image_gray);
    waitKey (0);
  }
}

