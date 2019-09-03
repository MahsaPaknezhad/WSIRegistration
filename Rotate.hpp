/*
 * Rotate.hpp
 *
 *  Created on: Jan 20, 2016
 *      Author: lohsy
 *  Improved on: Apr 01, 2018
 *	Author: mahsa
 *
 */

#ifndef ROTATE_HPP_
#define ROTATE_HPP_

#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <stdio.h>
#include "Utils.hpp"

using namespace cv;
/**
 * Takes 2 consecutive slices and finds
 * 1) rotation angle
 * 2) translation (minor adjustment)
 */
class Rotate {
public:

	Rotate(cv::Mat slice1);
	Rotate(cv::Mat slice1, cv::Mat slice2, cv::Mat mask1, cv::Mat mask2);

	void saveRotatedImage(const std::string filename);
	void saveRotationInformation(const std::string filename);
	void saveScoreInformation(const std::string filename);

	static int getAngleFromFile(const std::string filename);
	int getAngle();
	static cv::Point2i getTranslationFromFile(const std::string filename);
	cv::Point2i getTranslation();

private:
	cv::Mat slice1, slice2, warpedImage, warpedMask;
	cv::Mat mask1, mask2;
	int angle;
	cv::Point2i translate;
	//store the rotation information/score
	std::vector<cv::Vec4d> store;

	void findAngle();
	//find the squared difference in pixel intensity
	double getScore(cv::Mat image1, cv::Mat image2, cv::Mat mask1, cv::Mat mask2);
};

Rotate::Rotate(cv::Mat slice1, cv::Mat slice2, cv::Mat mask1, cv::Mat mask2) {
	this->slice1 = slice1;
	this->slice2 = slice2;
	this->mask1 = mask1;
	this->mask2 = mask2;

	//Grayscale
	cvtColor(slice1, this->slice1, CV_BGR2GRAY);
	cvtColor(slice2, this->slice2, CV_BGR2GRAY);
	cvtColor(mask1, this->mask1, CV_BGR2GRAY);
	cvtColor(mask2, this->mask2, CV_BGR2GRAY);

	findAngle();

	std::cout << "Translate: " << translate << std::endl;
	std::cout << "Rotation: " << angle << std::endl;
}

void Rotate::saveRotatedImage(const std::string filename) {
	imwrite(filename, warpedImage);
}



void Rotate::saveRotationInformation(const std::string filename) {
	std::ofstream outfile;
	outfile.open(filename.c_str());
	outfile << "===================================\n";
	outfile << "COMPUTER GENERATED DO NOT MODIFY\n";
	outfile << "===================================\n";
	outfile << "T" << " " << translate.x << " " << translate.y << "\n";
	outfile << "R" << " " << angle << " " << "\n";
	outfile.close();
}

void Rotate::saveScoreInformation(const std::string filename) {
	std::ofstream outfile;
	outfile.open(filename.c_str());
	for (cv::Vec4d p : store) {
		outfile << p.val[0] << " " << p.val[1] << " " << p.val[2] << " "
				<< p.val[3] << "\n";
	}
	outfile.close();
}

void Rotate::findAngle() {
	double minScore = std::numeric_limits<double>::max();
	cv::Scalar fill = slice2.at<uchar>(0, 0);
	cv::Scalar loDiff(0, 0, 0);
	cv::Scalar upDiff(fill[0] - 1, 0, 0);

	for (int i = -10; i <= 10; i += 1) {
		std::cout << ". ";
		for (int j = -10; j <= 10; j += 1) {
			for (int k = -14; k <= 14; k += 3) { //24

				//get matrix
				cv::Mat t = Utils::getTranslationMatrix(i, j);
				cv::Mat r = Utils::getRotationMatrix(k,
						cv::Point(slice2.cols / 2, slice2.rows / 2));
				cv::Mat finalMatrix;
				finalMatrix = r * t;
				cv::Mat affineMat = Utils::getAffineMatrixFrom3X3Matrix(
						finalMatrix);

				//warp
				cv::Mat warped;
				cv::warpAffine(slice2, warped, affineMat, slice2.size());
				Utils::floodCorners(warped, fill, loDiff, upDiff);
				cv::Mat warpedMask;
				cv::warpAffine(mask2, warpedMask, affineMat, mask2.size());

				//get score
				double score = getScore(slice1, warped, mask1, warpedMask);
				store.push_back(cv::Vec4d(i, j, k, score));

				//compare
				if (minScore > score) {
					minScore = score;
					translate.x = i;
					translate.y = j;
					angle = k;
				}
			}
		}
	}
	std::cout << "\n";

	//warp using best parameters found
	//get matrix
	cv::Mat t = Utils::getTranslationMatrix(translate.x, translate.y);
	cv::Mat r = Utils::getRotationMatrix(angle,
			cv::Point(slice2.cols / 2, slice2.rows / 2));
	cv::Mat finalMatrix;
	finalMatrix = r * t;
	cv::Mat affineMat = Utils::getAffineMatrixFrom3X3Matrix(finalMatrix);

	//warp
	cv::warpAffine(slice2, warpedImage, affineMat, slice2.size());
	Utils::floodCorners(warpedImage, fill, loDiff, upDiff);
}

int Rotate::getAngleFromFile(const std::string filename) {
	std::ifstream infile;
	infile.open(filename.c_str());
	std::string line;
	int angle;
	for (int i = 0; i < 4; i++) {
		std::getline(infile, line);
	}
	if (!(infile >> line >> angle)) {
		std::cout << "Error reading params from file" << std::endl;
	}
	return angle;
}

int Rotate::getAngle() {
	return angle;
}

cv::Point2i Rotate::getTranslationFromFile(const std::string filename) {
	std::ifstream infile;
	infile.open(filename.c_str());
	std::string line;
	int tx, ty;
	for (int i = 0; i < 3; i++) {
		std::getline(infile, line);
	}
	if (!(infile >> line >> tx >> ty)) {
		std::cout << "Error reading params from file" << std::endl;
	}
	return cv::Point2i(tx, ty);
}

cv::Point2i Rotate::getTranslation() {
	return translate;
}

double Rotate::getScore(cv::Mat image1, cv::Mat image2, cv::Mat mask1, cv::Mat mask2) {
	cv::Mat int1, int2, int3;
	image1.convertTo(int1, CV_64F);
	image2.convertTo(int2, CV_64F);

	cv::Mat combinedMasks;
	combinedMasks = mask1 + mask2;
	combinedMasks.convertTo(int3, CV_64F);
	cv::threshold(int3, int3, 254, 1, CV_THRESH_BINARY_INV);
	cv::Mat diff, diff2, sqr;
	diff = int1 - int2;
	diff2 = diff.mul(int3);
	sqr = diff2.mul(diff2, 1);
	cv::Scalar score = sum(sqr);

	return score[0];
}
#endif /* ROTATE_HPP_ */
