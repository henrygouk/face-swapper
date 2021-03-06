#ifndef _FACESWAPPER_HPP_
#define _FACESWAPPER_HPP_

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

enum DisplayMode
{
	faceSwap,
	boundingBox
};

class Face
{
	public:
		int misdetect;
		cv::Rect face;
		cv::Rect eyes;
		cv::Rect nose;
		cv::Rect mouth;
		cv::KalmanFilter kalmanFilter;

		Face(cv::Rect f);
		void predict();
		void update(cv::Rect inputPos, cv::Rect eyePos, cv::Rect nosePos, cv::Size frameSize);
};

class FaceSwapper
{
	public:
		void init();
		bool running();
		void processInput();
		void update();
		void draw();

	private:
		DisplayMode mMode;
		bool mRunning;
		cv::VideoCapture mCapture;
		cv::Mat mAlphaMask;
		cv::Mat mFrame;
		cv::Mat mGFrame;
		cv::CascadeClassifier mFaceDetector;
		cv::CascadeClassifier mEyeDetector;
		cv::CascadeClassifier mNoseDetector;
		cv::CascadeClassifier mMouthDetector;
		//std::vector<cv::Rect> mFaces;
		//std::vector<size_t> mMisdetect;
		std::vector<Face> mFaces;

		void detectNewFaces();
		void trackExistingFaces();
		void swapFaces();
};

#endif
