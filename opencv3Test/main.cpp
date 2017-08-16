#include "opencv2\opencv.hpp"
#include <iostream>
#include "myFunction.h"

typedef unsigned char uchar;

using namespace std;
using namespace cv;

void onMouse(int event, int x, int y, int flags, void* param);

const double a = 320.0 / tan(CV_PI / 6);


int main() {

	VideoCapture video;
	video.open(0);
	namedWindow("Original");

	//ColorDetector* cd = new ColorDetector();

	//cd->setMaxDistance(100);
	//cd->setTarget(0, 0, 255);

	MaskFilter* mf = new MaskFilter();
	//int mask[][3] = { { 1,2,1 },{ 0,0,0 },{ -1,-2,-1 } };
	Mat mask = (Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	Mat mask2 = (Mat_<int>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
	int mask3[] = { 1,1,1,1,1,1,1,1,1 };
	
	CannyEdge* canny = new CannyEdge();

	Morphology* morph = new Morphology();
	morph->setMask(mask3, 3);

	while (true) {
		Mat frame;
		video >> frame;
		imshow("Original", frame);

		setMouseCallback("Original", onMouse, reinterpret_cast<void*>(&frame));

		Mat red(Mat::zeros(frame.rows, frame.cols, CV_8UC1));
		Mat asd(Mat::zeros(frame.rows, frame.cols, CV_8UC1));
		Mat edge(Mat::zeros(frame.rows, frame.cols, CV_8UC1));

		//cd->detectColor(frame, red);

		//cvtColor(frame, red, CV_RGB2GRAY);
		//cvtColor(frame, asd, CV_RGB2GRAY);
		//mf->setMask(mask);
		//mf->applyMask(frame, red);
		//mf->setMask(mask2);
		//mf->applyMask(frame, asd);
		
		/*for (int j = 1; j < asd.rows - 1; ++j)
		{
			const uchar* previous = asd.ptr<uchar>(j - 1);
			const uchar* current = asd.ptr<uchar>(j);
			const uchar* next = asd.ptr<uchar>(j + 1);

			uchar* output = red.ptr<uchar>(j);

			for (int i = 1; i < (asd.cols - 1); ++i)
			{
				int cnt = 0;
				
				if (saturate_cast<uchar>(previous[i - 1] + previous[i + 1] + 2 * previous[i] - next[i + 1] - next[i - 1] - 2 * next[i]) > 80)
					*output++ = 255;
				else
					*output++ = 0;
			}
		}*/

		//canny->canny(frame, red, 80, 30);
		cvtColor(frame, asd, CV_RGB2GRAY);

		//Canny(frame, asd, 70, 50);

		threshold(asd, red, 127, 255, CV_THRESH_BINARY);
		morph->dilate(red, asd);
		morph->erode(red, edge);

		//filter2D(asd, asd, red.depth(), mask);

		imshow("test", red-edge);
		imshow("asd", asd);
		
		if (waitKey(1) == 27) {
			break;
		}
	}


	return 0;
}

void onMouse(int event, int x, int y, int flags, void* param) {
	Mat * im = reinterpret_cast<Mat*>(param);

	switch (event) {
	case CV_EVENT_LBUTTONDOWN:
		//cout << "click" << endl;
		int row = im->rows, col = im->cols;
		int nx = x - col / 2, ny = row / 2 - y;
		cout << nx << ", " << ny << "            " << (nx*180)/(a*CV_PI) << endl;
		break;
	}
}

void Sharpen(const Mat& myImage, Mat& Result)
{
	CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images

	Result.create(myImage.size(), myImage.type());
	const int nChannels = myImage.channels();


	for (int j = 1; j < myImage.rows - 1; ++j)
	{
		const uchar* previous = myImage.ptr<uchar>(j - 1);
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j + 1);

		uchar* output = Result.ptr<uchar>(j);

		for (int i = nChannels; i < nChannels * (myImage.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(5 * current[i]
				- current[i - nChannels] - current[i + nChannels] - previous[i] - next[i]);
		}
	}

	Result.row(0).setTo(Scalar(0));
	Result.row(Result.rows - 1).setTo(Scalar(0));
	Result.col(0).setTo(Scalar(0));
	Result.col(Result.cols - 1).setTo(Scalar(0));
}