#include "opencv2\opencv.hpp"
#include <stack>
#include <utility>

using namespace cv;
using namespace std;

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

int ColorDetector::getDistanceToTarget() {
	return abs((*target)[0] - (*value)[0]) + abs((*target)[1] - (*value)[1]) + abs((*target)[2] - (*value)[2]);
}

void ColorDetector::detectColor(Mat& image, Mat& result) {
	Mat_<Vec3b>::const_iterator it = image.begin<Vec3b>();
	Mat_<Vec3b>::const_iterator itend = image.end<Vec3b>();
	Mat_<uchar>::iterator itout = result.begin<uchar>();

	for (; it != itend; ++it, ++itout) {
		if (getDistance(*it) <= maxDistance) {
			*itout = 255;
		}
		else {
			*itout = 0;
		}
	}
}

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

void MaskFilter::applyMask(Mat image, Mat result) {
	int offset = dimension / 2;
	Mat gray(image.size(), CV_8UC1);
	cvtColor(image, gray, CV_RGB2GRAY);
	//uchar* data = (uchar*)result.data;
	int* maskData = (int*)mask.data;
	//cout << mask << endl;
	for (int j = offset; j < gray.rows - offset; ++j)
	{
		const uchar* previous = gray.ptr<uchar>(j - offset);
		const uchar* current = gray.ptr<uchar>(j);
		const uchar* next = gray.ptr<uchar>(j + offset);


		uchar* output = result.ptr<uchar>(j);

		for (int i = offset; i < (gray.cols - offset); ++i)
		{
			uchar temp = saturate_cast<uchar>(maskData[0] * previous[i - 1] + maskData[1] * previous[i] + maskData[2] * previous[i + 1] +
				maskData[3] * current[i - 1] + maskData[4] * current[i] + maskData[5] * current[i + 1] +
				maskData[6] * next[i - 1] + maskData[7] * next[i] + maskData[8] * next[i + 1]);

			if (temp > 80)
				*output++ = temp;
			else
				*output++ = 0;
		}
	}

}

class CannyEdge {
public:
	CannyEdge() {
	}
	void canny(Mat input, Mat output, int tHigh, int tLow);
	inline void gaussianBlur(Mat input);
	void setThres(int tHigh, int tLow) { this->tHigh = tHigh; this->tLow = tLow; }
	void setDimension(int width, int height) { this->width = width; this->height = height; }
	//void setInput(Mat input) { this->input = input; }
	//void setOutput(Mat output) { this->output = output; }
	

private:
	const Mat gaussianMask = (Mat_<double>(5, 5) << 2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2) / 159.0;
	const int sobelMaskX[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	const int sobelMaskY[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	int tHigh, tLow;
	Mat mag, ang;
	//Mat input, output;
	int width, height;
	stack< pair<int, int> > st;
};

inline void CannyEdge::gaussianBlur(Mat input) {
	filter2D(input, input, input.depth(), gaussianMask);
}

void CannyEdge::canny(Mat input, Mat output, int tHigh, int tLow) {

	if (input.channels() != 1) cvtColor(input, input, CV_RGB2GRAY);
	if (input.size() != output.size()) output = Mat(input.size(), CV_8UC1);

	setDimension(input.cols, input.rows);
	//setInput(input);
	//setOutput(output);

	setThres(tHigh, tLow);
	gaussianBlur(input);

	//cv::imshow("gaussian", input);

	mag = Mat(input.size(), CV_64FC1);
	ang = Mat(input.size(), CV_8UC1);

	for (int j = 1; j < height - 1; j++) {
		const uchar* pre = input.ptr<uchar>(j - 1);
		const uchar* curr = input.ptr<uchar>(j);
		const uchar* next = input.ptr<uchar>(j + 1);

		const uchar* rows[3] = { pre, curr, next };

		double* pMag = mag.ptr<double>(j);
		uchar* pAng = ang.ptr<uchar>(j);

		//uchar* result = output.ptr<uchar>(j);

		for (int i = 1; i < width - 1; i++) {
			double sumX = 0.0;
			double sumY = 0.0;

			for (int a = 0; a < 3; a++) {
				for (int b = 0; b < 3; b++) {
					sumX += sobelMaskX[3 * a + b] * rows[a][i + b - 1];
					sumY += sobelMaskY[3 * a + b] * rows[a][i + b - 1];
				}
			}
			pMag[i] = sqrt(sumX*sumX + sumY*sumY);
			double theta;
			if (pMag[i] == 0) {
				if (sumY == 0)
					theta = 0;
				else
					theta = 90;
			}
			else {
				theta = atan2((float)sumY, (float)sumX)*180.0 / CV_PI;
			}

			if ((theta > -22.5 && theta < 22.5) || theta > 157.5 || theta < -157.5)
				pAng[i] = 0;
			else if ((theta >= 22.5 && theta < 67.5) || (theta >= -157.5 && theta < -112.5))
				pAng[i] = 45;
			else if ((theta >= 67.5 && theta <= 112.5) || (theta >= -112.5 && theta <= -67.5))
				pAng[i] = 90;
			else pAng[i] = 135;
		}
	}
	Mat imageCand = Mat::zeros(input.size(), CV_8UC1);
	for (int j = 1; j < height - 1; j++) {

		uchar* pCand = imageCand.ptr<uchar>(j);
		double* pMag_pre = mag.ptr<double>(j - 1);
		double* pMag = mag.ptr<double>(j);
		double* pMag_next = mag.ptr<double>(j + 1);
		uchar* pAng = ang.ptr<uchar>(j);

		for (int i = 1; i < width - 1; i++) {
			switch (pAng[i]) {
			case 0:
				if (pMag[i] > pMag[i - 1] && pMag[i] > pMag[i + 1]) {
					pCand[i] = 255;
				}
				break;
			case 45:
				if (pMag[i] > pMag_pre[i + 1] && pMag[i] > pMag_next[i - 1])
					pCand[i] = 255;
				break;
			case 90:
				if (pMag[i] > pMag_pre[i] && pMag[i] > pMag_next[i])
					pCand[i] = 255;
				break;
			case 135:
				if (pMag[i] > pMag_pre[i - 1] && pMag[i] > pMag_next[i + 1])
					pCand[i] = 255;
				break;
			}
		}
	}
	//cv::imshow("Candidate", imageCand);

	for (int j = 1; j < height - 1; j++) {

		uchar* pOut = output.ptr<uchar>(j);
		uchar* pCand = imageCand.ptr<uchar>(j);
		double* pMag_pre = mag.ptr<double>(j - 1);
		double* pMag = mag.ptr<double>(j);
		double* pMag_next = mag.ptr<double>(j + 1);
		uchar* pAng = ang.ptr<uchar>(j);

		for (int i = 1; i < width - 1; i++) {
			if (pCand[i] != 0 && pOut[i] != 255) {
				if (pMag[i] > tHigh) {
					//pOut[i] = 255;
					this->st.push( (pair<int, int>)make_pair(i, j) );
				}
				/*
				else if (pMag[i] > tLow) {
					switch (pAng[i]) {
					case 0:
						if (pMag_pre[i] > tHigh || pMag_next[i] > tHigh) {
							pOut[i] = 255;
						}
						break;
					case 45:
						if (pMag_pre[i - 1] > tHigh || pMag_next[i + 1] > tHigh)
							pOut[i] = 255;
						break;
					case 90:
						if (pMag[i - 1] > tHigh || pMag[i + 1] > tHigh)
							pOut[i] = 255;
						break;
					case 135:
						if (pMag_pre[i + 1] > tHigh || pMag_next[i - 1] > tHigh)
							pOut[i] = 255;
						break;
					}
				}*/

				while(!(this->st.empty())) {
					//if ((this->st.empty())) break;

					pair<int, int> pa;
					pa = (this->st.top());
					this->st.pop();
					int a = pa.first, b = pa.second;
					if (a == 0 || b == 0 || a == width - 1 || b == height - 1) {
					}
					else {

						uchar* pOut_pre2 = output.ptr<uchar>(b - 1);
						uchar* pOut2 = output.ptr<uchar>(b);
						uchar* pOut_next2 = output.ptr<uchar>(b + 1);
						double* pMag_pre2 = mag.ptr<double>(b - 1);
						double* pMag2 = mag.ptr<double>(b);
						double* pMag_next2 = mag.ptr<double>(b + 1);
						uchar* pAng2 = ang.ptr<uchar>(b);

						if (pOut2[a] != 255) {

							pOut2[a] = 255;

							switch (pAng2[a]) {
							case 0:
								if (pMag_pre2[a] > tLow || pMag_next2[a] > tLow) {
									pOut_pre2[a] = pOut_next2[a] = 255;
									st.push(make_pair(a, b - 1));
									st.push(make_pair(a, b + 1));
								}
								break;
							case 45:
								if (pMag_pre2[a - 1] > tLow || pMag_next2[a + 1] > tLow) {
									pOut_pre2[a - 1] = pOut_next2[a + 1] = 255;
									st.push(make_pair(a - 1, b - 1));
									st.push(make_pair(a + 1, b + 1));
								}
								break;
							case 90:
								if (pMag2[a - 1] > tLow || pMag2[a + 1] > tLow) {
									pOut2[a - 1] = pOut2[a + 1] = 255;
									st.push(make_pair(a - 1, b));
									st.push(make_pair(a + 1, b));
								}
								break;
							case 135:
								if (pMag_pre2[a + 1] > tLow || pMag_next2[a - 1] > tLow) {
									pOut_pre2[a + 1] = pOut_next2[a - 1] = 255;
									st.push(make_pair(a + 1, b - 1));
									st.push(make_pair(a - 1, b + 1));
								}
								break;
							}
						}

					}

				}
			}
		}
	}

}

class Morphology {
public :
	Morphology() {};
	void setMask(int mask[], int dimension) {
		for (int i = 0; i < dimension*dimension; i++) { this->mask[i] = mask[i]; }
		this->dimension = dimension; 
	}
	void dilate(Mat input, Mat output);
	void erode(Mat input, Mat output);
	//void dilate(Mat input, Mat output, Mat mask);
	//void erode(Mat input, Mat output, Mat mask);

private :
	int mask[100];
	int dimension;
};

void Morphology::dilate(Mat input, Mat output) {

	int offset = dimension / 2;
	int width = input.cols;
	int height = input.rows;
	for (int j = offset; j < height - offset; j++) {
		uchar* pPre = input.ptr<uchar>(j - 1);
		uchar* pCurr = input.ptr<uchar>(j);
		uchar* pNext = input.ptr<uchar>(j + 1);
		uchar* pInput[3] = { pPre, pCurr, pNext };

		uchar* pOut = output.ptr<uchar>(j);

		for (int i = offset; i < width - offset; i++) {
			int max = 0;
			for (int k = 0; k < 3; k++) {
				for (int w = 0; w < 3; w++) {

					int temp = pInput[k][i - 1 + w] + mask[k*dimension + w];

					if (temp > max)
						max = temp;
				}
			}

			pOut[i] = saturate_cast<uchar>(max);

		}
	}
}

void Morphology::erode(Mat input, Mat output) {

	int offset = dimension / 2;
	int width = input.cols;
	int height = input.rows;
	for (int j = offset; j < height - offset; j++) {
		uchar* pPre = input.ptr<uchar>(j - 1);
		uchar* pCurr = input.ptr<uchar>(j);
		uchar* pNext = input.ptr<uchar>(j + 1);
		uchar* pInput[3] = { pPre, pCurr, pNext };

		uchar* pOut = output.ptr<uchar>(j);

		for (int i = offset; i < width - offset; i++) {
			int min = 255;
			for (int k = 0; k < 3; k++) {
				for (int w = 0; w < 3; w++) {
					int temp = pInput[k][i - 1 + w] - mask[k*dimension + w];
					if (temp < min)
						min = temp;
				}
			}

			pOut[i] = saturate_cast<uchar>(min);

		}
	}
}