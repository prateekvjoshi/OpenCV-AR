// OpenCV Augmented Reality
// Author: Prateek Joshi

#include "cv.hpp"
#include "highgui.hpp"
#include "imgproc.hpp"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 360

#define CHESSBOARD_WIDTH 7
#define CHESSBOARD_HEIGHT 6

int main()
{
    VideoCapture capture(0);
    Mat image, image_big, frame;
    int key = 0;
    
	if ( !capture.isOpened() )
    {
        cerr << "ERROR: Unable to capture frames from webcam." << endl;
        return -1;
    }
    
    Mat overlayPic = imread("tiger.jpg");
    
    if(overlayPic.empty())
    {
        cerr << "Unable to read overlay pic. Check if path to the variable 'overlayPic' is specified correctly.\n" << endl;
        return -1;
    }
    
    // The chessboard pattern actually has 7x6 squares, which means it has 6x5 = 30 enclosed corners. Hence the picture will be overlaid on top of this 6x5 inner rectangle.
	Size overlay_size(CHESSBOARD_WIDTH-1, CHESSBOARD_HEIGHT-1);
    
    // This vector will hold all the detected corners on the chessboard pattern
	vector<Point2f> corners;
    
    // Loop until the user presses 'q' or 'Q'
	while(key!='q' && key!='Q')
	{
        // Capture the next frame from the camera
		capture >> image_big;
        
        // Input image resized to a fixed size to work with different input sources
        Size dsize(WINDOW_WIDTH, WINDOW_HEIGHT);
        image.create(WINDOW_HEIGHT, WINDOW_WIDTH, image_big.type());
        resize(image_big, image, dsize);
        flip(image, image, 1);
        
        Mat disp(image.rows, image.cols, image.type());
        Mat copy_image(image.rows, image.cols, image.type());
        Mat neg_image(image.rows, image.cols, image.type());
        
		Mat gray;
        cvtColor(image, gray, CV_BGR2GRAY);
        
        // Detect the chessboard pattern. Returns 1 if present.
        bool flag = findChessboardCorners(image, overlay_size, corners);
        
		if(flag == TRUE)
	    {            
            // This function identifies the pattern from the gray image, saves the valid group of corners
            cornerSubPix(gray, corners, Size(11,11), Size(-1,-1), TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
        
            vector<Point2f> p;
            vector<Point2f> q;
                
            Mat blank;
            blank.create(overlayPic.rows, overlayPic.cols, overlayPic.type());
            blank = Scalar(0);
            bitwise_not(blank,blank);
				
            // Set of source points to calculate the transformation matrix
            q.push_back(Point2f(0,0));
            q.push_back(Point2f(overlayPic.cols,0));
            q.push_back(Point2f(overlayPic.cols, overlayPic.rows));
            q.push_back(Point2f(0, overlayPic.rows));
            
            // Set of detected corners. Since it is a 7x6 pattern, 30 enclosed corners will be detected. We have to take the four corners corresponding to the corners of the 6x5 inner rectangle. In the 'corners' vector, points are arranged in a row-major fashion (0-indexed).
            p.push_back(corners[0]);
            p.push_back(corners[CHESSBOARD_WIDTH-2]);
            p.push_back(corners[(CHESSBOARD_WIDTH-1)*(CHESSBOARD_HEIGHT-1)-1]);
            p.push_back(corners[(CHESSBOARD_WIDTH-1)*(CHESSBOARD_HEIGHT-2)]);
				
            // Compute the transformation matrix
            Mat warp_matrix = getPerspectiveTransform(q, p);
                
            // Boolean operations to perform augmentation. The detected pattern will be replaced by the desired pic, without affecting other parts of the input video.
			neg_image = Scalar(0);
			copy_image = Scalar(0);
                
			warpPerspective(overlayPic, neg_image, warp_matrix, Size(neg_image.cols, neg_image.rows));
            warpPerspective(blank, copy_image, warp_matrix, Size(copy_image.cols, neg_image.rows));
            bitwise_not(copy_image, copy_image);

            bitwise_and(copy_image, image, copy_image);
            bitwise_or(copy_image, neg_image, image);
        }

        imshow("Video", image);
        
		key = cvWaitKey(10);
		
	}
    
    destroyAllWindows();
    
    return 1;
}