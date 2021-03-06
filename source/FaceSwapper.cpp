#include "FaceSwapper.hpp"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>

using namespace cv;
using namespace std;

float clamp(float a, float b, float c)
{
	return max(a, min(b, c));
}

cv::Rect doubleRectSize(const cv::Rect &inputRect, const cv::Size &frameSize)
{
    cv::Rect outputRect;
    // Double rect size
    outputRect.width = inputRect.width * 2;
    outputRect.height = inputRect.height * 2;

    // Center rect around original center
    outputRect.x = inputRect.x - inputRect.width / 2;
    outputRect.y = inputRect.y - inputRect.height / 2;

    // Handle edge cases
    if (outputRect.x < 0) {
        outputRect.width += outputRect.x;
        outputRect.x = 0;
    }   
    if (outputRect.y < 0) {
        outputRect.height += outputRect.y;
        outputRect.y = 0;
    }   

    if (outputRect.x + outputRect.width > frameSize.width) {
        outputRect.width = frameSize.width - outputRect.x;
    }   
    if (outputRect.y + outputRect.height > frameSize.height) {
        outputRect.height = frameSize.height - outputRect.y;
    }   

    return outputRect;
}

void overlayImage(Mat* src, Mat* overlay, const Point& location)
{
    for (int y = max(location.y, 0); y < src->rows; ++y)
    {
        int fY = y - location.y;

        if (fY >= overlay->rows)
            break;

        for (int x = max(location.x, 0); x < src->cols; ++x)
        {
            int fX = x - location.x;

            if (fX >= overlay->cols)
                break;

            double opacity = ((double)overlay->data[fY * overlay->step + fX * overlay->channels() + overlay->channels() - 1]) / 255;

            for (int c = 0; opacity > 0 && c < src->channels(); ++c)
            {
                unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
                unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
                src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
            }
        }
    }
}

Face::Face(Rect facePos)
{
	face = facePos;
	misdetect = 0;

	//Track 2d positions and velocities of face, mouth, eyes, and nose (16 componenets in total)
	//Only want estimates of the positions (8 components total)
	//kalmanFilter = KalmanFilter(16, 8, 0, CV_32F);

	//Track faces eyes and nose
	kalmanFilter = KalmanFilter(18, 9, 0);

	kalmanFilter.transitionMatrix = *(Mat_<float>(18, 18) <<
			1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
			0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
			0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
			0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
			0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,
			0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,
			0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,
			0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,
			0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
			0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
			0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
			0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
			0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1);

	kalmanFilter.statePost.at<float>(0) = face.x + face.width / 2;
	kalmanFilter.statePost.at<float>(1) = face.y + face.height / 2;
	kalmanFilter.statePost.at<float>(2) = face.width;
	kalmanFilter.statePost.at<float>(3) = face.x + face.width / 2;
	kalmanFilter.statePost.at<float>(4) = face.y + face.height / 3;
	kalmanFilter.statePost.at<float>(5) = ((float)face.width / 3.0) * 2.0;
	kalmanFilter.statePost.at<float>(6) = face.x + face.width / 2;
	kalmanFilter.statePost.at<float>(7) = face.y + face.height / 2;
	kalmanFilter.statePost.at<float>(8) = ((float)face.width / 6.0);
	kalmanFilter.statePost.at<float>(9) = 0;
	kalmanFilter.statePost.at<float>(10) = 0;
	kalmanFilter.statePost.at<float>(11) = 0;
	kalmanFilter.statePost.at<float>(12) = 0;
	kalmanFilter.statePost.at<float>(13) = 0;
	kalmanFilter.statePost.at<float>(14) = 0;
	kalmanFilter.statePost.at<float>(15) = 0;
	kalmanFilter.statePost.at<float>(16) = 0;
	kalmanFilter.statePost.at<float>(17) = 0;

	setIdentity(kalmanFilter.measurementMatrix);
	setIdentity(kalmanFilter.processNoiseCov, Scalar::all(1));
	setIdentity(kalmanFilter.measurementNoiseCov, Scalar::all(10));
	setIdentity(kalmanFilter.errorCovPost, Scalar::all(10));
}

void Face::predict()
{
	kalmanFilter.predict();
}

void Face::update(Rect inputPos, Rect eyePos, Rect nosePos, Size frameSize)
{
	misdetect = 0;

	Mat measurement(9, 1, CV_32F);
	measurement.at<float>(0) = inputPos.x + inputPos.width / 2;
	measurement.at<float>(1) = inputPos.y + inputPos.height / 2;
	measurement.at<float>(2) = inputPos.width;
	measurement.at<float>(3) = eyePos.x + eyePos.width / 2;
	measurement.at<float>(4) = eyePos.y + eyePos.height / 2;
	measurement.at<float>(5) = eyePos.width;
	measurement.at<float>(6) = nosePos.x + nosePos.width / 2;
	measurement.at<float>(7) = nosePos.y + nosePos.height / 2;
	measurement.at<float>(8) = nosePos.width;

	Mat est = kalmanFilter.correct(measurement);

	face.width = est.at<float>(2);
	face.height = est.at<float>(2);
	face.x = est.at<float>(0) - face.width / 2;
	face.y = est.at<float>(1) - face.height / 2;

	eyes.width = est.at<float>(5);
	eyes.height = ((float)eyePos.height / (float)eyePos.width) * est.at<float>(5);
	eyes.x = est.at<float>(3) - eyes.width / 2;
	eyes.y = est.at<float>(4) - eyes.height / 2;

	nose.width = est.at<float>(8);
	nose.height = ((float)nosePos.height / (float)nosePos.width) * est.at<float>(8);
	nose.x = est.at<float>(6) - nose.width / 2;
	nose.y = est.at<float>(7) - nose.height / 2;

	face.x = clamp(0, face.x, frameSize.width - face.width);
	face.y = clamp(0, face.y, frameSize.height - face.height);
}

void FaceSwapper::init()
{
	mCapture = VideoCapture(-1);
	mCapture.set(CV_CAP_PROP_FRAME_WIDTH, 960);
	mCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 540);
	mRunning = true;
	mMode = DisplayMode::faceSwap;

	mFaceDetector.load("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml");
	mEyeDetector.load("/usr/share/opencv/haarcascades/haarcascade_mcs_eyepair_small.xml");
	mMouthDetector.load("/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml");
	mNoseDetector.load("/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml");
	mAlphaMask = imread("../alphamask.png", IMREAD_UNCHANGED);
}

bool FaceSwapper::running()
{
	return mRunning;
}

void FaceSwapper::processInput()
{
	int c = waitKey(1);

	if((char)c == 'q')
	{
		mRunning = false;
	}
	else if((char)c == 'b')
	{
		mMode = DisplayMode::boundingBox;
	}
	else if((char)c == 's')
	{
		mMode = DisplayMode::faceSwap;
	}
	else if((char)c == 'w')
	{
		auto t = std::time(nullptr);
	    auto tm = *std::localtime(&t);
	    stringstream ss;
		ss << put_time(&tm, "%d-%m-%Y-%H-%M-%S.png");

		imwrite(ss.str(), mFrame);
	}

	mCapture.read(mFrame);
	flip(mFrame, mFrame, 1);
	cvtColor(mFrame, mGFrame, CV_BGR2GRAY);
	equalizeHist(mGFrame, mGFrame);
}

void FaceSwapper::update()
{
	static int frameId = 0;

	if(frameId % 150 == 0)
	{
		detectNewFaces();
	}

	frameId++;

	trackExistingFaces();

	if(mMode == DisplayMode::faceSwap)
	{
		swapFaces();
	}
}

void FaceSwapper::draw()
{
	Mat fullFrame(Size(1920, 1080), CV_8UC3);
	resize(mFrame, fullFrame, fullFrame.size());

	imshow("Face Swapper", fullFrame);
}

/**
  Searches through the entire current frame for all faces. mFaces is updated with
  any ROIs that do not overlap significantly with already tracked faces.
  */
void FaceSwapper::detectNewFaces()
{
	//Detect all the faces in the current frame
	//mFaceDetector.detectMultiScale(mGFrame, mFaces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
	
	/*mMisdetect.clear();

	for(size_t i = 0; i < mFaces.size(); i++)
	{
		mMisdetect.push_back(3);
	}*/

	vector<Rect> faces;
	mFaceDetector.detectMultiScale(mGFrame, faces, 1.1, 5, CV_HAAR_SCALE_IMAGE, Size(30, 30));

	mFaces.clear();

	for(size_t i = 0; i < faces.size(); i++)
	{
		mFaces.push_back(Face(faces[i]));
	}
}

float euclideanDistance(Rect a, Rect b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

/**
  Updates the bounding boxes for detected faces in order to compensate for movement.
  Also does temporal smoothing to compensate for ocassional false negatives.
  */
void FaceSwapper::trackExistingFaces()
{
	//Iterate over each tracked face
	for(size_t i = 0; i < mFaces.size(); i++)
	{
		mFaces[i].predict();

		//Run the face detector on a the face ROI
		std::vector<Rect> faces;
		auto roi = doubleRectSize(mFaces[i].face, mGFrame.size());
		mFaceDetector.detectMultiScale(mGFrame(roi), faces, 1.1, 5, CV_HAAR_SCALE_IMAGE, Size(roi.width * 4 / 10, roi.height * 4 / 10), Size(roi.width * 6 / 10, roi.width * 6 / 10));

		if(faces.size() > 0)
		{	
			Rect inputPos = faces[0];
			inputPos.x += roi.x;
			inputPos.y += roi.y;

			std::vector<Rect> eyes;
			std::vector<Rect> noses;
			//std::vector<Rect> mouths;
			mEyeDetector.detectMultiScale(mGFrame(inputPos), eyes, 1.1, 3, CV_HAAR_SCALE_IMAGE);
			mNoseDetector.detectMultiScale(mGFrame(inputPos), noses, 1.1, 2, CV_HAAR_SCALE_IMAGE);
			//mMouthDetector.detectMultiScale(mGFrame(roi), mouths, 1.1, 5, CV_HAAR_SCALE_IMAGE);

			if(eyes.size() == 0 || noses.size() == 0)
			{
				continue;
			}

			Rect eyePos = eyes[0];
			float eyeDist = euclideanDistance(eyePos, mFaces[i].eyes);

			for(size_t j = 1; j < eyes.size(); j++)
			{
				float d = euclideanDistance(eyes[j], mFaces[i].eyes);

				if(d < eyeDist)
				{
					eyePos = eyes[j];
				}
			}

			eyePos.x += inputPos.x;
			eyePos.y += inputPos.y;

			Rect nosePos = noses[0];
			nosePos.x += inputPos.x;
			nosePos.y += inputPos.y;

			mFaces[i].update(inputPos, eyePos, nosePos, mFrame.size());

			if(mMode == DisplayMode::boundingBox)
			{
				rectangle(mFrame, mFaces[i].face, CV_RGB(255, 0, 255));
				rectangle(mFrame, mFaces[i].eyes, CV_RGB(255, 0, 0));
				rectangle(mFrame, mFaces[i].nose, CV_RGB(0, 255, 0));

				/*
				for(size_t j = 0; j < mouths.size(); j++)
				{
					rectangle(mFrame, Rect(roi.x + mouths[j].x, roi.y + mouths[j].y, mouths[j].width, mouths[j].height), CV_RGB(0,255,0));
				}

				for(size_t j = 0; j < noses.size(); j++)
				{
					rectangle(mFrame, Rect(roi.x + noses[j].x, roi.y + noses[j].y, noses[j].width, noses[j].height), CV_RGB(0,0,255));
				}

				for(size_t j = 0; j < eyes.size(); j++)
				{
					rectangle(mFrame, Rect(roi.x + eyes[j].x, roi.y + eyes[j].y, eyes[j].width, eyes[j].height), CV_RGB(255,0,0));
				}
				*/
			}
		}
		else
		{
			//if(mMisdetect[i] == 3)
			if(mFaces[i].misdetect == 5)
			{
				mFaces.erase(mFaces.begin() + i);
				//mMisdetect.erase(mMisdetect.begin() + i);
				i--;
			}
			else
			{
				//mMisdetect[i]++;
				mFaces[i].misdetect++;
			}
		}
	}
}

void computeMean(const Mat &input, float *output, float *sd)
{
	output[0] = 0;
	output[1] = 0;
	output[2] = 0;
	sd[0] = 0;
	sd[1] = 0;
	sd[2] = 0;

	for(size_t j = 0; j < input.rows; j++)
	{
		for(size_t i = 0; i < input.cols; i++)
		{
			Vec3b p = input.at<Vec3b>(i, j);
			output[0] += p[0];
			output[1] += p[1];
			output[2] += p[2];
		}
	}

	output[0] /= input.size().height * input.size().width;
	output[1] /= input.size().height * input.size().width;
	output[2] /= input.size().height * input.size().width;

	for(size_t j = 0; j < input.rows; j++)
	{
		for(size_t i = 0; i < input.cols; i++)
		{
			Vec3b p = input.at<Vec3b>(i, j);
			sd[0] += pow(output[0] - p[0], 2);
			sd[1] += pow(output[1] - p[1], 2);
			sd[2] += pow(output[2] - p[2], 2);
		}
	}

	sd[0] /= input.size().height * input.size().width;
	sd[1] /= input.size().height * input.size().width;
	sd[2] /= input.size().height * input.size().width;

	sd[0] = sqrt(sd[0]);
	sd[1] = sqrt(sd[1]);
	sd[2] = sqrt(sd[2]);
}

void changeMean(Mat &input, float *oldMean, float *newMean, float *oldSd, float *newSd)
{
	float bs = newSd[0] / oldSd[0];
	float gs = newSd[1] / oldSd[1];
	float rs = newSd[2] / oldSd[2];

	for(size_t j = 0; j < input.rows; j++)
	{
		for(size_t i = 0; i < input.cols; i++)
		{
			Vec3b p = input.at<Vec3b>(i, j);
			float b = p[0];
			float g = p[1];
			float r = p[2];
			/*b *= bs;
			g *= gs;
			r *= rs;
			b += newMean[0] - oldMean[0];
			g += newMean[1] - oldMean[1];
			r += newMean[2] - oldMean[2];*/
			//b /= (oldMean[0] / newMean[0]);
			//g /= (oldMean[1] / newMean[1]);
			//r /= (oldMean[2] / newMean[2]);
			b -= oldMean[0];
			g -= oldMean[1];
			r -= oldMean[2];
			b *= bs;
			g *= gs;
			r *= rs;
			b += newMean[0];
			g += newMean[1];
			r += newMean[2];
			p[0] = clamp(0, b, 255);
			p[1] = clamp(0, g, 255);
			p[2] = clamp(0, r, 255);
			input.at<Vec3b>(i, j) = p;
		}
	}
}

void transferColor(Mat &primary, const Mat &secondary)
{
	Mat plab = primary;//.clone();
	Mat slab = secondary;//.clone();

	cvtColor(primary, plab, CV_BGR2Lab);
	cvtColor(secondary, slab, CV_BGR2Lab);

	float pMean[3];
	float sMean[3];
	float pSd[3];
	float sSd[3];

	computeMean(plab, &pMean[0], &pSd[0]);
	computeMean(slab, &sMean[0], &sSd[0]);
	changeMean(plab, &pMean[0], &sMean[0], &pSd[0], &sSd[0]);

	cvtColor(plab, primary, CV_Lab2BGR);
	cvtColor(slab, secondary, CV_Lab2BGR);
}

/**
  Swaps faces!
  */
void FaceSwapper::swapFaces()
{
	Mat frameDup = mFrame.clone();
	int fromTo[] = {3, 3};

	for(size_t i = 0; i < mFaces.size(); i++)
	{
		auto cf = mFaces[i].face;
		auto nf = mFaces[(i + 1) % mFaces.size()].face;
		auto currentFace = frameDup(cf);
		auto nextFace = mFrame(nf);
		Mat currentFaceBGRA(nextFace.size(), CV_8UC4);
		Mat resizedCurrentFace(nextFace.size(), CV_8UC3); 
		Mat alphaMask(nextFace.size(), CV_8UC4);
		
		resize(currentFace, resizedCurrentFace, resizedCurrentFace.size());
		resize(mAlphaMask, alphaMask, alphaMask.size());
		
		transferColor(resizedCurrentFace, nextFace);

		cvtColor(resizedCurrentFace, currentFaceBGRA, CV_BGR2BGRA, 4);
		mixChannels(&alphaMask, 1, &currentFaceBGRA, 1, fromTo, 1);
		overlayImage(&nextFace, &currentFaceBGRA, Point());
	}
}
