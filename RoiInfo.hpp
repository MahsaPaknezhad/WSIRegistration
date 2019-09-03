/*
 * RoiInfo.hpp
 *
 *  Created on: Jan 20, 2016
 *      Author: lohsy
 */

#ifndef ROIINFO_HPP_
#define ROIINFO_HPP_

#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <vector>
#include <limits>
#include <string>
#include <fstream>
#include "Utils.hpp"



typedef std::vector<cv::Point> Contour;
typedef std::vector<Contour> Contours;
std::vector<std::vector<Point> >hull;


class RoiInfo {

public:
	RoiInfo(cv::Mat image, std::string sampleName, bool log);
	void saveImages(const std::string dir);
	void saveRoiInformation(const std::string filename);
	cv::Point3i getRoiInformation();
	static cv::Point3i getRoiInformationFromFile(std::string filename);
	void saveRoiMaskImage(const std::string filename);

private:
	std::string sampleName;
	Contours contours;
	cv::Mat image, gray, blurred, thresholded, closed, opened, edge, maskedImage, overlay, overlayHull;
	std::vector<int> contourIndices;
	bool log;

	void preprocess();
	void findContours();
	void findBestContourAndROI();
	void findConvexHullOfBestContours();


	cv::Mat thresholdImage();
	cv::Point2f getContourCenter(Contour contour);
	void searchHoughSpace(cv::Mat selectedContourEdgeImage);

	//returns the number of pixels near (dist < 1 pixel) a circle of radius r centere at (i,j)
	//OPTIMIZATION - dont sqrt, use contours instead of searching for edge pixel
	int getEdgePixelsNearCircle(int i, int j, int r,
			cv::Mat selectedContourEdgeImage);
	//faster version but gives slight difference in hough circle results
	int getEdgePixelsNearCircle(int i, int j, int r);

	//mask image using selected contour
	void maskImage(const Contours contours, const std::vector<int> contourIndices,
			const cv::Mat inputImage, cv::Mat &outputImage);
	//mask image using MBC (minimum bounding circle) or hough circle
	void maskImage(const cv::Point2f centre, const float radius,
			const cv::Mat inputImage, cv::Mat &outputImage);

	void drawOverlay();
};

RoiInfo::RoiInfo(cv::Mat image, std::string sampleName, bool log) {
	image.copyTo(this->image);
	this->sampleName = sampleName;
	this->log = log;

	preprocess();
	findContours();
	findBestContourAndROI();
	findConvexHullOfBestContours();
	//mask using contour, optional, minor differences
	maskImage(contours, contourIndices, this->image, maskedImage);

	drawOverlay();
}

void RoiInfo::saveImages(const std::string dir) {

	cv::Mat saveImage;

	cv::resize(maskedImage, saveImage, cv::Size(maskedImage.cols/2, maskedImage.rows/2));
		imwrite(dir + "clean.tiff", saveImage);

		if(log){
			cv::resize(gray, saveImage, cv::Size(gray.cols/2, gray.rows/2));
			imwrite(dir + "gray.tiff", saveImage);

			cv::resize(blurred, saveImage, cv::Size(blurred.cols/2, blurred.rows/2));
			imwrite(dir + "blurred.tiff", saveImage);

			cv::resize(thresholded, saveImage, cv::Size(thresholded.cols/2, thresholded.rows/2));
			imwrite(dir + "thresholded.tiff", saveImage);

			cv::resize(edge, saveImage, cv::Size(edge.cols/2, edge.rows/2));
			imwrite(dir + "edge.tiff", saveImage);

			cv::resize(closed, saveImage, cv::Size(closed.cols/2, closed.rows/2));
			imwrite(dir + "close.tiff", saveImage);

			cv::resize(opened, saveImage, cv::Size(opened.cols/2, opened.rows/2));
			imwrite(dir + "open.tiff", saveImage);

			cv::resize(overlay, saveImage, cv::Size(overlay.cols/2, overlay.rows/2));
			imwrite(dir + "overlay.tiff", saveImage);

			cv::resize(overlayHull, saveImage, cv::Size(overlayHull.cols/2, overlayHull.rows/2));
			imwrite(dir + "overlay_hull.tiff", saveImage);
		}
}

void RoiInfo::saveRoiInformation(const std::string filename) {
	std::ofstream outfile;
	outfile.open(filename.c_str());
	outfile << "===================================\n";
	outfile << "COMPUTER GENERATED DO NOT MODIFY\n";
	outfile << "===================================\n";
    for( int i = 0 ; i < contourIndices.size(); i++){
		Contour c = contours[contourIndices.at(i)];
		cv::Point2d contourCenter = getContourCenter(c);
		outfile << "CONTOUR" << " " << contourCenter.x << " " << contourCenter.y
		<< "\n";
		outfile << c.size() << " ";
		for (cv::Point p : c) {
			outfile << p.x << " " << p.y << " ";
		}
		outfile << "\n";
    }
	outfile.close();
}

void RoiInfo::saveRoiMaskImage(const std::string filename) {
	cv::Mat contourMask;
	contourMask = cv::Mat::zeros(gray.size(), CV_8UC3);
	drawContours(contourMask, hull, 0, cv::Scalar(255, 255, 255),-1);

	cv::Mat saveImage;
	cv::resize(contourMask, saveImage, cv::Size(gray.cols/2, gray.rows/2));
	imwrite(filename, saveImage);
}



cv::Point3i RoiInfo::getRoiInformationFromFile(std::string filename) {
	std::ifstream infile;
	infile.open(filename.c_str());
	std::string line;
	int x, y, r;
	for (int i = 0; i < 5; i++) {
		std::getline(infile, line);
	}
	while (infile >> line >> x >> y >> r) {

	}
	cv::Point3i p(x, y, r);
	return p;
}

void RoiInfo::preprocess() {
	cvtColor(image, gray, CV_BGR2GRAY);
	cv::GaussianBlur(gray, blurred, cv::Size(0, 0), 10, 10);
	thresholded = thresholdImage();
	int morph_size = 20;
	cv::Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(-1,-1) );
	morphologyEx( thresholded, closed, 3, element );
	morphologyEx( closed, opened, 2, element );
	//erode( closed, eroded, element );
}

void RoiInfo::findContours() {
	std::vector<cv::Vec4i> hierarchy;
	cv::Canny(opened, edge, 0.04, 1, 3, true);
	cv::findContours(edge, contours, hierarchy, CV_RETR_TREE,
			CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void RoiInfo::findBestContourAndROI() {

	//eclipse complains about min and max, they are ok.
	double minArea = (image.cols * image.rows)/20;
	double maxDistance = 1.5 * std::max(double(image.cols), double(image.rows));
	double largestArea = std::numeric_limits<double>::min();
	int index = -1;
	double largestAreaDistance;

	for (unsigned int i = 0; i < contours.size(); i++) {

		cv::Point2f currentCenter;
		Contour contoursPoly;
		approxPolyDP(cv::Mat(contours[i]), contoursPoly, 3, true);
		cv::RotatedRect boundRect = minAreaRect(cv::Mat(contoursPoly));
		double currentArea = boundRect.size.width * boundRect.size.height;
		currentCenter = boundRect.center;
		double currentDistance = sqrt(
				(image.cols / 2 - currentCenter.x)
						* (image.cols / 2 - currentCenter.x)
						+ (image.rows / 2 - currentCenter.y)
								* (image.rows / 2 - currentCenter.y));

		if (currentArea > minArea && currentDistance < maxDistance) {
			contourIndices.push_back(i);
		}
		if (currentArea > largestArea && currentDistance < largestAreaDistance) {
			largestAreaDistance = currentDistance;
			largestArea = currentArea;
			index = i;
		}
	}

	if ( contourIndices.size() == 0 && index != -1)
		contourIndices.push_back(index);
}

cv::Point2f RoiInfo::getContourCenter(Contour contour) {
	cv::Moments m = cv::moments(contour, false);
	return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
}


void RoiInfo::findConvexHullOfBestContours(){
	hull.resize(1);
	Contour contour;
	contour.resize(0);
	for( int i = 0; i < contourIndices.size(); i++ ){
		Contour temp = contours[contourIndices.at(i)];
		contour.insert(contour.end(),temp.begin(), temp.end());
	}
    convexHull(Mat(contour), hull[0], false);

}


void RoiInfo::maskImage(const Contours contours, const std::vector<int>  contourIndices,
		const cv::Mat inputImage, cv::Mat &outputImage) {
	cv::Mat contourMask;
	contourMask = cv::Mat::zeros(inputImage.size(), CV_8UC3);

	//draw selected contour (white) on "contourMask", -1 indicates filled contour.
	drawContours(contourMask, hull, 0, cv::Scalar(255, 255, 255), -1);


	//mask the image, the removed parts are black in colour
	bitwise_and(inputImage, contourMask, outputImage);
	//flood fill the black parts with white

	floodFill(outputImage, cv::Point(0, 0), cv::Scalar(255, 255, 255));
}

void RoiInfo::maskImage(const cv::Point2f centre, const float radius,
		const cv::Mat inputImage, cv::Mat &outputImage) {
	cv::Mat mask;
	mask = cv::Mat::zeros(inputImage.size(), CV_8UC3);

	//create a circle on image "mask" (white), -1 indicates filled contour.
	circle(mask, centre, (int) radius, cv::Scalar(255, 255, 255), -1);
	//mask the image, the removed parts are black in colour
	bitwise_and(inputImage, mask, outputImage);
	//flood fill the black parts with white
	floodFill(outputImage, cv::Point(0, 0), cv::Scalar(255, 255, 255));
}

cv::Mat RoiInfo::thresholdImage() {
	cv::Scalar meanValue = cv::mean(blurred);
	uchar m = meanValue[0];

	cv::Mat t;
	t = cv::Mat::zeros(blurred.size(), blurred.type());
	for (int i = 0; i < blurred.rows; i++) {
		const uchar* bpt = blurred.ptr<uchar>(i);
		uchar* tpt = t.ptr<uchar>(i);
		for (int j = 0; j < blurred.cols; j++) {
			double s = bpt[j];
			if (s > m)
				tpt[j] = 255;
			else
				tpt[j] = 0;
		}
	}

	return t;
}

void RoiInfo::drawOverlay() {
	image.copyTo(overlay);
	for (int i=0 ; i < contourIndices.size(); i++){
	    drawContours(overlay, contours, contourIndices.at(i), cv::Scalar(255, 0, 0), 15);
	}
	image.copyTo(overlayHull);
	drawContours(overlayHull, hull, 0, cv::Scalar(255, 0, 0), 15, 8, std::vector<Vec4i>(), 0, Point() );

}

#endif /* RoiInfo_HPP_ */
