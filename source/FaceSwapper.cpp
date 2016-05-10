#include "FaceSwapper.hpp"

#include <algorithm>
#include <iostream>

using namespace cv;
using namespace std;

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

void FaceSwapper::init()
{
	mCapture = VideoCapture(-1);
	//mCapture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	//mCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	mRunning = true;

	mFaceDetector.load("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml");
	mEyeDetector.load("/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
	mAlphaMask = imread("../alphamask.png", IMREAD_UNCHANGED);
}

bool FaceSwapper::running()
{
	return mRunning;
}

void FaceSwapper::processInput()
{
	mCapture.read(mFrame);
	flip(mFrame, mFrame, 1);

	int c = waitKey(1);

	if((char)c == 'q')
	{
		mRunning = false;
	}
}

void FaceSwapper::update()
{
	static int frameId = 0;

	if(frameId % 30 == 0)
	{
		detectNewFaces();
	}

	frameId++;

	trackExistingFaces();
	swapFaces();
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
	cvtColor(mFrame, mGFrame, CV_BGR2GRAY);
	equalizeHist(mGFrame, mGFrame);
	mFaceDetector.detectMultiScale(mGFrame, mFaces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
	
	mMisdetect.clear();

	for(size_t i = 0; i < mFaces.size(); i++)
	{
		mMisdetect.push_back(3);
	}
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
		//Run the face detector on a the face ROI
		std::vector<Rect> faces;
		std::vector<Rect> eyes;
		auto roi = doubleRectSize(mFaces[i], mGFrame.size());
		mFaceDetector.detectMultiScale(mGFrame(roi), faces, 1.1, 5, 0, Size(roi.width * 4 / 10, roi.height * 4 / 10), Size(roi.width * 6 / 10, roi.width * 6 / 10));

		if(faces.size() > 0)
		{
			//mEyeDetector.detectMultiScale(mGFrame(faces[0]), eyes, 1.1, 1, CV_HAAR_SCALE_IMAGE);

			//if(eyes.size() > 0)
			{
				mFaces[i] = faces[0];
				mFaces[i].x += roi.x;
				mFaces[i].y += roi.y;
				mMisdetect[i] = 0;

				continue;
			}
		}

		if(mMisdetect[i] == 3)
		{
			mFaces.erase(mFaces.begin() + i);
			mMisdetect.erase(mMisdetect.begin() + i);
			i--;
		}
		else
		{
			mMisdetect[i]++;
		}
	}

	cout << "Faces: " << mFaces.size() << endl;
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

float clamp(float a, float b, float c)
{
	return max(a, min(b, c));
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
		auto cf = mFaces[i];
		auto nf = mFaces[(i + 1) % mFaces.size()];
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
