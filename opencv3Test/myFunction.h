#pragma once
#include "opencv2\opencv.hpp"
#include <stack>

using namespace cv;

class ColorDetector {
public:
	ColorDetector() { this->maxDistance = 60; }
	ColorDetector(Vec3b* target, int dist = 60) { (this->target) = target; this->maxDistance = dist; }

	void setTarget(uchar blue, uchar green, uchar red) { (this->target) = new Vec3b(blue, green, red); }
	void setMaxDistance(int dist) { this->maxDistance = dist; }
	void setValue(uchar blue, uchar green, uchar red) { (this->value) = new Vec3b(blue, green, red); }

	int getDistance(Vec3b value) { setValue(value[0], value[1], value[2]); return getDistanceToTarget(); }

	void detectColor(Mat& image, Mat& result);

private:
	int maxDistance;
	Vec3b* target = NULL;
	Vec3b* value = NULL;
	int getDistanceToTarget();

};



class MaskFilter {
public:
	MaskFilter() { dimension = 3; mask.create(dimension, dimension, CV_8UC1); }
	void applyMask(Mat image, Mat result);
	void setDimension(int dimension) { this->dimension = dimension; mask.create(dimension, dimension, CV_8UC1); }
	void setMask(Mat mask) { (this->mask) = mask; }


private:
	Mat mask;
	int dimension;

};

class CannyEdge {
public:
	CannyEdge() {
	}
	void canny(Mat input, Mat output, int tHigh, int tLow);
	inline void gaussianBlur(Mat input);
	void setThres(int tHigh, int tLow) { this->tHigh = tHigh; this->tLow = tLow; }

private:
	const Mat gaussianMask = (Mat_<double>(5, 5) << 2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2) / 159.0;
	const int sobelMaskX[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	const int sobelMaskY[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	int tHigh, tLow;
	Mat mag;
	Mat ang;
};

class Morphology {
public:
	Morphology() {};
	void setMask(int mask[], int dimension) {
		for (int i = 0; i < dimension*dimension; i++) { this->mask[i] = mask[i]; }
		this->dimension = dimension;
	}
	void dilate(Mat input, Mat output);
	void erode(Mat input, Mat output);
	//void dilate(Mat input, Mat output, Mat mask);
	//void erode(Mat input, Mat output, Mat mask);

private:
	int mask[100];
	int dimension;
};