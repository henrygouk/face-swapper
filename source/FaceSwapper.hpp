#ifndef _FACESWAPPER_HPP_
#define _FACESWAPPER_HPP_

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class FaceSwapper
{
	public:
		void init();
		bool running();
		void processInput();
		void update();
		void draw();

	private:
		bool mRunning;
		cv::VideoCapture mCapture;
		cv::Mat mAlphaMask;
		cv::Mat mFrame;
		cv::CascadeClassifier mFaceDetector;
		cv::CascadeClassifier mEyeDetector;
		std::vector<cv::Rect> mFaces;
		std::vector<size_t> mMisdetect;

		void detectNewFaces();
		void trackExistingFaces();
		void swapFaces();
};

#endif
