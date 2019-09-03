/*
 * Utils.hpp
 *
 *  Created on: Jan 20, 2016
 *      Author: lohsy
 *  Improved on: Apr 01, 2018
 *	Author: mahsa
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cv.h>
#include <cv.hpp>
#include <highgui.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <imgproc.hpp>
#include <imgcodecs.hpp>
#include <videoio.hpp>
#include <core.hpp>

/**
 * Utility functions
 */
class Utils {
public:
	static void showImage(const cv::Mat image, const std::string name);
	static bool isSame(const cv::Mat image1, const cv::Mat image2);
	static bool fileExist(const std::string filename);

	static cv::Mat getTranslationMatrix(const int tx, const int ty);
	//opencv implementation is scaled with respect to (0,0).
	//we want to scale with respect to center of image.
	static cv::Mat getScaleMatrix(const double sx, const double sy,
			const cv::Point p);
	static cv::Mat getRotationMatrix(const int angle, const cv::Point center);
	static cv::Mat getAffineMatrixFrom3X3Matrix(const cv::Mat mat);
	static cv::Mat get3X3MatrixFromAffineMatrix(const cv::Mat mat);

	static void floodCorners(cv::Mat &image, cv::Scalar fill, cv::Scalar loDiff,
			cv::Scalar upDiff);

	static cv::Mat getDownSizedImage(const cv::Mat image);

	static cv::Mat loadAffineMatrixFromFile(const std::string filename);
	static void saveAffineMatrixToFile(const cv::Mat affineMatrix,
			const std::string filename);
	static void saveAffineMatrixToFile(
			const std::vector<cv::Mat> affineMatrixes,
			const std::string filename);
	static void saveAffineMatrixCostToFile(const double affineMatrixCost,
			const std::string filename);

	static const int draw_shift_bits = 4;
	static const int draw_multiplier = 1 << draw_shift_bits;
	static void drawKeyPointsCustom(const cv::Mat& image,
			const std::vector<cv::KeyPoint>& keypoints, cv::Mat& outImage,
			const cv::Scalar& _color, int flags, int lw);
	static void drawKeypointCustom(cv::Mat& img, const cv::KeyPoint& p,
			const cv::Scalar& color, int flags, int lw);
	static void drawMatchesCustom(const cv::Mat& img1,
			const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& img2,
			const std::vector<cv::KeyPoint>& keypoints2,
			const std::vector<cv::DMatch>& matches1to2, cv::Mat& outImg,
			const cv::Scalar& matchColor, const cv::Scalar& singlePointColor,
			const std::vector<char>& matchesMask, int flags, int lw);
	static void prepareImgAndDrawKeypointsCustom(const cv::Mat& img1,
			const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& img2,
			const std::vector<cv::KeyPoint>& keypoints2, cv::Mat& outImg,
			cv::Mat& outImg1, cv::Mat& outImg2,
			const cv::Scalar& singlePointColor, int flags, int lw);

	static void saveROIToFile(const cv::Mat ROI, const std::string filename);
	static cv::Mat loadROIFromFile(const std::string filename);

};

void Utils::showImage(const cv::Mat image, const std::string name) {

	cv::namedWindow(name, CV_WINDOW_NORMAL);
	cv::imshow(name, image);
	cv::waitKey(0);

}

bool Utils::fileExist(const std::string filename) {
	std::ifstream infile;
	infile.open(filename.c_str());
	if (infile.good()) {
		infile.close();
		return true;
	} else {
		infile.close();
		return false;
	}
}

cv::Mat Utils::getTranslationMatrix(const int tx, const int ty) {
	cv::Mat mat = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
	mat.at<double>(0, 0) = 1;
	mat.at<double>(1, 1) = 1;
	mat.at<double>(2, 2) = 1;
	mat.at<double>(0, 2) = tx;
	mat.at<double>(1, 2) = ty;
	return mat;
}

cv::Mat Utils::getScaleMatrix(const double sx, const double sy,
		const cv::Point p) {
	cv::Mat mat = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
	mat.at<double>(0, 0) = sx;
	mat.at<double>(1, 1) = sy;
	mat.at<double>(2, 2) = 1;
	mat.at<double>(0, 2) = -(sx * p.x - p.x);
	mat.at<double>(1, 2) = -(sy * p.y - p.y);
	return mat;
}

cv::Mat Utils::getRotationMatrix(const int angle, const cv::Point center) {
	cv::Mat mat = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
	cv::Mat affineMat = cv::getRotationMatrix2D(center, angle, 1);
	mat.at<double>(0, 0) = affineMat.at<double>(0, 0);
	mat.at<double>(0, 1) = affineMat.at<double>(0, 1);
	mat.at<double>(1, 0) = affineMat.at<double>(1, 0);
	mat.at<double>(1, 1) = affineMat.at<double>(1, 1);
	mat.at<double>(0, 2) = affineMat.at<double>(0, 2);
	mat.at<double>(1, 2) = affineMat.at<double>(1, 2);
	mat.at<double>(2, 2) = 1;
	return mat;
}

cv::Mat Utils::getAffineMatrixFrom3X3Matrix(const cv::Mat mat) {
	cv::Mat affineMatrix(mat, cv::Range(0, 2), cv::Range(0, 3));
	return affineMatrix;
}

cv::Mat Utils::get3X3MatrixFromAffineMatrix(const cv::Mat mat) {
	cv::Mat affineMatrix = cv::Mat::zeros(3, 3, CV_64F);
	affineMatrix.at<double>(0, 0) = mat.at<double>(0, 0);
	affineMatrix.at<double>(0, 1) = mat.at<double>(0, 1);
	affineMatrix.at<double>(1, 0) = mat.at<double>(1, 0);
	affineMatrix.at<double>(1, 1) = mat.at<double>(1, 1);
	affineMatrix.at<double>(0, 2) = mat.at<double>(0, 2);
	affineMatrix.at<double>(1, 2) = mat.at<double>(1, 2);
	affineMatrix.at<double>(2, 2) = 1;
	return affineMatrix;
}

void Utils::floodCorners(cv::Mat &image, cv::Scalar fill, cv::Scalar loDiff =
		cv::Scalar(), cv::Scalar upDiff = cv::Scalar()) {
	floodFill(image, cv::Point(0, 0), fill, 0, loDiff, upDiff);
	floodFill(image, cv::Point(0, image.rows - 1), fill, 0, loDiff, upDiff);
	floodFill(image, cv::Point(image.cols - 1, 0), fill, 0, loDiff, upDiff);
	floodFill(image, cv::Point(image.cols - 1, image.rows - 1), fill, 0, loDiff,
			upDiff);
}

cv::Mat Utils::getDownSizedImage(const cv::Mat image) {
	cv::Mat downsize;

	cv::resize(image, downsize, cv::Size(image.cols / 2, image.rows / 2));

	return downsize;
}

cv::Mat Utils::loadAffineMatrixFromFile(const std::string filename) {
	cv::Mat affineMat = cv::Mat::zeros(cv::Size(3, 2), CV_64FC1);
	std::ifstream i;
	i.open(filename.c_str());

	if (i.good()) {
		std::string line;
		std::getline(i, line);
		std::getline(i, line);
		std::getline(i, line);

		double a, b, c, d, e, f;
		i >> a >> b >> c >> d >> e >> f;
		affineMat.at<double>(0, 0) = a;
		affineMat.at<double>(0, 1) = b;
		affineMat.at<double>(0, 2) = c;
		affineMat.at<double>(1, 0) = d;
		affineMat.at<double>(1, 1) = e;
		affineMat.at<double>(1, 2) = f;
	} else {
		throw std::invalid_argument(filename + " does not exist");
	}
	i.close();

	return affineMat;
}

void Utils::saveAffineMatrixCostToFile(const double affineMatrixCost,const std::string filename) {
	std::ofstream o;

	o.open(filename.c_str());
	o << "===================================\n";
	o << "COMPUTER GENERATED DO NOT MODIFY\n";
	o << "===================================\n";
	o << affineMatrixCost << "\n";
	o.close();
}

void Utils::saveAffineMatrixToFile(const cv::Mat affineMatrix,
		const std::string filename) {
	std::ofstream o;

	o.open(filename.c_str());
	o << "===================================\n";
	o << "COMPUTER GENERATED DO NOT MODIFY\n";
	o << "===================================\n";
	o << affineMatrix.at<double>(0, 0) << " " << affineMatrix.at<double>(0, 1)
			<< " " << affineMatrix.at<double>(0, 2) << "\n"
			<< affineMatrix.at<double>(1, 0) << " "
			<< affineMatrix.at<double>(1, 1) << " "
			<< affineMatrix.at<double>(1, 2) << "\n";
	o.close();
}

void Utils::saveROIToFile(const cv::Mat ROI, const std::string filename) {
	std::ofstream o;

	o.open(filename.c_str());
	o << "===================================\n";
	o << "COMPUTER GENERATED DO NOT MODIFY\n";
	o << "===================================\n";
	o << ROI.at<double>(0, 0) << " " << ROI.at<double>(0, 1) << "\n";
	o << ROI.at<double>(1, 0) << " " << ROI.at<double>(1, 1) << "\n";
	o.close();
}

cv::Mat Utils::loadROIFromFile(const std::string filename) {
	cv::Mat ROI = cv::Mat::zeros(cv::Size(2, 2), CV_64FC1);
	std::ifstream i;
	i.open(filename.c_str());

	if (i.good()) {
		std::string line;
		std::getline(i, line);
		std::getline(i, line);
		std::getline(i, line);

		double a, b, c, d;
		i >> a >> b;
		i >> c >> d;
		ROI.at<double>(0, 0) = a;
		ROI.at<double>(0, 1) = b;
		ROI.at<double>(1, 0) = c;
		ROI.at<double>(1, 1) = d;
	} else {
		throw std::invalid_argument(filename + " does not exist");
	}
	i.close();

	return ROI;
}

void Utils::saveAffineMatrixToFile(const std::vector<cv::Mat> affineMatrixes,
		const std::string filename) {
	std::ofstream o;

	o.open(filename.c_str());
	o << "===================================\n";
	o << "COMPUTER GENERATED DO NOT MODIFY\n";
	o << "===================================\n";
	for (cv::Mat affineMat : affineMatrixes) {
		o << affineMat.at<double>(0, 0) << " " << affineMat.at<double>(0, 1)
				<< " " << affineMat.at<double>(0, 2) << " "
				<< affineMat.at<double>(1, 0) << " "
				<< affineMat.at<double>(1, 1) << " "
				<< affineMat.at<double>(1, 2) << std::endl;
	}
	o.close();
}

bool Utils::isSame(const cv::Mat image1, const cv::Mat image2) {

	if (!image1.data || !image2.data) {
		std::cout << "image1 or image2 is not loaded" << std::endl;
		return false;
	}

	if (image1.rows != image2.rows)
		return false;

	if (image1.cols != image2.cols)
		return false;

	for (int x = 0; x < image1.rows; x++) {
		const uchar* pt1 = image1.ptr<uchar>(x);
		const uchar* pt2 = image2.ptr<uchar>(x);
		for (int y = 0; y < image2.cols; y++) {
			if (pt1[y] != pt2[y]) {
				return false;
			}
		}
	}
	return true;
}

void Utils::drawKeyPointsCustom(const cv::Mat& image,
		const std::vector<cv::KeyPoint>& keypoints, cv::Mat& outImage,
		const cv::Scalar& _color, int flags, int lw) {
	if (!(flags & cv::DrawMatchesFlags::DRAW_OVER_OUTIMG)) {
		if (image.type() == CV_8UC3) {
			image.copyTo(outImage);
		} else if (image.type() == CV_8UC1) {
			cvtColor(image, outImage, CV_GRAY2BGR);
		} else {
			CV_Error( CV_StsBadArg, "Incorrect type of input image.\n");
		}
	}

	cv::RNG& rng = cv::theRNG();
	bool isRandColor = _color == cv::Scalar::all(-1);

	CV_Assert( !outImage.empty());
	std::vector<cv::KeyPoint>::const_iterator it = keypoints.begin(), end =
			keypoints.end();
	for (; it != end; ++it) {
		cv::Scalar color =
				isRandColor ? cv::Scalar(rng(256), rng(256), rng(256)) : _color;
		Utils::drawKeypointCustom(outImage, *it, color, flags, lw);
	}
}

void Utils::drawKeypointCustom(cv::Mat& img, const cv::KeyPoint& p,
		const cv::Scalar& color, int flags, int lw) {
	CV_Assert( !img.empty());
	cv::Point center(cvRound(p.pt.x * draw_multiplier),
			cvRound(p.pt.y * draw_multiplier));

	if (flags & cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS) {
		int radius = cvRound(p.size / 2 * draw_multiplier); // KeyPoint::size is a diameter

		// draw the circles around keypoints with the keypoints size
		circle(img, center, radius, color, lw, CV_AA, draw_shift_bits);

		// draw orientation of the keypoint, if it is applicable
		if (p.angle != -1) {
			float srcAngleRad = p.angle * (float) CV_PI / 180.f;
			cv::Point orient(cvRound(cos(srcAngleRad) * radius),
					cvRound(sin(srcAngleRad) * radius));
			line(img, center, center + orient, color, lw, CV_AA,
					draw_shift_bits);
		}
#if 0
		else
		{
			// draw center with R=1
			int radius = 1 * draw_multiplier;
			circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
		}
#endif
	} else {
		// draw center with R=3
		int radius = 3 * draw_multiplier;
		circle(img, center, radius, color, lw, CV_AA, draw_shift_bits);
	}
}

void Utils::drawMatchesCustom(const cv::Mat& img1,
		const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& img2,
		const std::vector<cv::KeyPoint>& keypoints2,
		const std::vector<cv::DMatch>& matches1to2, cv::Mat& outImg,
		const cv::Scalar& matchColor, const cv::Scalar& singlePointColor,
		const std::vector<char>& matchesMask, int flags, int lw) {
	if (!matchesMask.empty() && matchesMask.size() != matches1to2.size())
		CV_Error( CV_StsBadSize,
				"matchesMask must have the same size as matches1to2");

	cv::Mat outImg1, outImg2;
	prepareImgAndDrawKeypointsCustom(img1, keypoints1, img2, keypoints2, outImg,
			outImg1, outImg2, singlePointColor, flags, lw);

	// draw matches
	for (size_t m = 0; m < matches1to2.size(); m++) {
		if (matchesMask.empty() || matchesMask[m]) {
			int i1 = matches1to2[m].queryIdx;
			int i2 = matches1to2[m].trainIdx;
			CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints1.size()));
			CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints2.size()));

			const cv::KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];

			cv::RNG& rng = cv::theRNG();
			bool isRandMatchColor = matchColor == cv::Scalar::all(-1);
			cv::Scalar color =
					isRandMatchColor ?
							cv::Scalar(rng(256), rng(256), rng(256)) :
							matchColor;

			drawKeypointCustom(outImg1, kp1, color, flags, lw);
			drawKeypointCustom(outImg2, kp2, color, flags, lw);

			cv::Point2f pt1 = kp1.pt, pt2 = kp2.pt, dpt2 = cv::Point2f(
					std::min(pt2.x + outImg1.cols, float(outImg.cols - 1)),
					pt2.y);

			line(outImg,
					cv::Point(cvRound(pt1.x * draw_multiplier),
							cvRound(pt1.y * draw_multiplier)),
					cv::Point(cvRound(dpt2.x * draw_multiplier),
							cvRound(dpt2.y * draw_multiplier)), color, lw,
					CV_AA, draw_shift_bits);
		}
	}
}

void Utils::prepareImgAndDrawKeypointsCustom(const cv::Mat& img1,
		const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& img2,
		const std::vector<cv::KeyPoint>& keypoints2, cv::Mat& outImg,
		cv::Mat& outImg1, cv::Mat& outImg2, const cv::Scalar& singlePointColor,
		int flags, int lw) {
	cv::Size size(img1.cols + img2.cols, MAX(img1.rows, img2.rows));
	if (flags & cv::DrawMatchesFlags::DRAW_OVER_OUTIMG) {
		if (size.width > outImg.cols || size.height > outImg.rows)
			CV_Error( CV_StsBadSize,
					"outImg has size less than need to draw img1 and img2 together");
		outImg1 = outImg(cv::Rect(0, 0, img1.cols, img1.rows));
		outImg2 = outImg(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
	} else {
		outImg.create(size, CV_MAKETYPE(img1.depth(), 3));
		outImg = cv::Scalar::all(0);
		outImg1 = outImg(cv::Rect(0, 0, img1.cols, img1.rows));
		outImg2 = outImg(cv::Rect(img1.cols, 0, img2.cols, img2.rows));

		if (img1.type() == CV_8U)
			cvtColor(img1, outImg1, CV_GRAY2BGR);
		else
			img1.copyTo(outImg1);

		if (img2.type() == CV_8U)
			cvtColor(img2, outImg2, CV_GRAY2BGR);
		else
			img2.copyTo(outImg2);
	}

	// draw keypoints
	if (!(flags & cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS)) {
		cv::Mat _outImg1 = outImg(cv::Rect(0, 0, img1.cols, img1.rows));
		Utils::drawKeyPointsCustom(_outImg1, keypoints1, _outImg1,
				singlePointColor,
				flags + cv::DrawMatchesFlags::DRAW_OVER_OUTIMG, lw);

		cv::Mat _outImg2 = outImg(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
		Utils::drawKeyPointsCustom(_outImg2, keypoints2, _outImg2,
				singlePointColor,
				flags + cv::DrawMatchesFlags::DRAW_OVER_OUTIMG, lw);
	}
}
#endif /* UTILS_HPP_ */
