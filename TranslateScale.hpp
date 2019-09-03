/*
 * TranslateScale.hpp
 *
*  Created on: Jan 20, 2016
 *      Author: lohsy
 *  Improved on: Apr 01, 2018
 *	Author: mahsa
 */

#ifndef TRANSLATESCALE_HPP_
#define TRANSLATESCALE_HPP_

#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include "Utils.hpp"

class TranslateScale {
public:

	TranslateScale(const cv::Point3i circleInfo1, const cv::Mat image);
	TranslateScale(const cv::Point3i circleInfo1, const cv::Point3i circleInfo2,
			const cv::Mat image);
	void saveTranslatedScaledImage(const std::string filename);
	void saveGrayBMPImage(const std::string filename);
	void saveTransformationInformation(const std::string filename);

	cv::Point2i getTranslation();
	static cv::Point2i getTranslationFromFile(const std::string filename);
	cv::Point2d getScaling();
	static cv::Point2d getScalingFromFile(const std::string filename);

private:
	cv::Point3i circleInfo1, circleInfo2;
	cv::Mat image, warped;

	cv::Point2i translate;
	cv::Point2d scale;
	cv::Mat affineMat;

	void getScalingParameters();
	void warpImage();
};

TranslateScale::TranslateScale(const cv::Point3i circleInfo1,
		const cv::Mat image) {
	image.copyTo(this->image);
	this->circleInfo1 = circleInfo1;

	translate = cv::Point2i(image.cols / 2 - circleInfo1.x,
			image.rows / 2 - circleInfo1.y);
	scale = cv::Point2d(1, 1);

	std::cout << "Translate: " << translate << std::endl;
	std::cout << "Scale: " << scale.x << " " << scale.y << std::endl;

	warpImage();
}

TranslateScale::TranslateScale(const cv::Point3i circleInfo1,
		const cv::Point3i circleInfo2, const cv::Mat image) {
	image.copyTo(this->image);
	this->circleInfo1 = circleInfo1;
	this->circleInfo2 = circleInfo2;

	translate = cv::Point2i(image.cols / 2 - circleInfo2.x,
			image.rows / 2 - circleInfo2.y);
	getScalingParameters();

	std::cout << "Translate: " << translate << std::endl;
	std::cout << "Scale: " << scale.x << " " << scale.y << std::endl;

	warpImage();
}

void TranslateScale::saveTranslatedScaledImage(const std::string filename) {
	imwrite(filename, warped);
}

void TranslateScale::saveGrayBMPImage(const std::string filename) {
	cv::Mat bmpImage;
	cv::resize(warped, bmpImage, cv::Size(warped.cols / 4, warped.rows / 4));
	cvtColor(bmpImage, bmpImage, CV_BGR2GRAY);
	imwrite(filename, bmpImage);
}

void TranslateScale::saveTransformationInformation(const std::string filename) {
	std::ofstream outfile;
	outfile.open(filename.c_str());
	outfile << "===================================\n";
	outfile << "COMPUTER GENERATED DO NOT MODIFY\n";
	outfile << "===================================\n";
	outfile << "T" << " " << translate.x << " " << translate.y << "\n";
	outfile << "S" << " " << scale.x << " " << scale.y << "\n";
	outfile.close();
}

cv::Point2i TranslateScale::getTranslation() {
	return translate;
}

cv::Point2i TranslateScale::getTranslationFromFile(const std::string filename) {
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

cv::Point2d TranslateScale::getScaling() {
	return scale;
}

cv::Point2d TranslateScale::getScalingFromFile(const std::string filename) {
	std::ifstream infile;
	infile.open(filename.c_str());
	std::string line;
	double sx, sy;
	for (int i = 0; i < 4; i++) {
		std::getline(infile, line);
	}
	if (!(infile >> line >> sx >> sy)) {
		std::cout << "Error reading params from file" << std::endl;
	}
	return cv::Point2d(sx, sy);
}

void TranslateScale::getScalingParameters() {
	scale = cv::Point2d((double) circleInfo1.z / (double) circleInfo2.z,
			(double) circleInfo1.z / (double) circleInfo2.z);
}
/**
 * [sx 0 tx
 *  0 sy ty]
 */
void TranslateScale::warpImage() {
	cv::Mat t = Utils::getTranslationMatrix(translate.x, translate.y);
	cv::Mat s = Utils::getScaleMatrix(scale.x, scale.y,
			cv::Point(image.cols / 2, image.rows / 2));
	cv::Mat finalMatrix;
	finalMatrix = s * t;
	affineMat = Utils::getAffineMatrixFrom3X3Matrix(finalMatrix);
	warpAffine(image, warped, affineMat, image.size());
	Utils::floodCorners(warped, cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0),
			cv::Scalar(254, 254, 254));
}

#endif /* TRANSLATESCALE_HPP_ */
