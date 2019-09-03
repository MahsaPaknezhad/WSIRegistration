/*
 * Rotate.hpp
 *
*  Created on: Jan 20, 2016
 *      Author: lohsy
 *  Improved on: Apr 01, 2018
 *	Author: mahsa
 *
 */
#include<string>
#include<iostream>
#include <stdio.h>
#include "RoiInfo.hpp"
#include "Utils.hpp"
#include "MumfordShah.hpp"
#include <opencv2/tracking.hpp>

#ifndef REGISTER_HPP_
#define REGISTER_HPP_

class Register {
public:
	Register(std::string inputDir, std::string sampleName, std::string id, double lambda, int initialSlice, int numImages, std::string mumfordAppLocation, std::string roiSelectionApp, bool log);
private:

	double lambda;
    bool log;
	std::string dir, sampleName;

	std::string ORIGINAL, MUMFORD, ALIGNED;
	std::string ROI_MASK_FILE, ROI_INFO_FILE, TRANSLATE_SCALE_FILE;
	std::string CLEAN, MUMFORD_APP_LOCATION, MUMFORD_LOG;
	std::string MUMFORD_ROTATED, Mask_rotated, ROTATION_INFO_FILE, ROTATE_LOG_FILE;
	std::string COARSE_RESULT, MASK_RESULT1, MASK_RESULT2, COARSE_AFFINE_MAT_FILE;
	std::string REGISTERED, SIFT_AFFINE_MAT_FILE;
	std::string SIFT_AFFINE_MAT_COST_FILE;
	std::string FINAL_AFFINE_MAT_FILE;
	std::string SIFT_AFFINE_MAT_LOG_FILE;
	std::string FINAL_AFFINE_MAT_ALL_SLICES_FILE;
	std::string ROI_LOCATION_FILE, ROI_SELECTION_APP;
	std::string MASK;
	std::string GRAY_COMMAND_1A, GRAY_COMMAND_1B, GRAY_COMMAND_2A;
	std::string GRAY_COMMAND_A1, GRAY_COMMAND_A2, GRAY_COMMAND_A3;
	std::string GRAY_COMMAND_B1;
	int REFERENCE_IMAGE_WIDTH, REFERENCE_IMAGE_HEIGHT;
	int NUM_IMAGES, INITIAL_SLICE;


	std::string workingDirSlice1;

	void start();
	void startSingle(int index);
	void startCoarse(int index, std::string workingDir);

	void startFine(int index, std::string workingDir);

	void doRoiInfo(int index, std::string workingDir);
	//first slice
	void doAlign(std::string workingDir);
	//other slices
	void doAlign(std::string workingDir1, std::string workingDir2);

	void prepareForMumford(std::string workingDir);
	//other slices
	void prepareForMumford(std::string workingDir1, std::string workingDir2);

	void doMumfordSegmentation(std::string workingDir);

	void doRotation(std::string workingDir1, std::string workingDir2, std::string maskName);

	void transformImageForSIFT(std::string workingDir, std::string maskName1, std::string maskName2, int index);

	void performSIFT(std::string workingDir1, std::string workingDir2);

	void performSIFTWithMask(std::string workingDir1, std::string workingDir2, std::string maskName);

	void saveFinalAffineMatrix(std::string workingDir);

	void saveAllSlicesFinalAffineMatrix();

	void setRegionOfInterest(std::string workingDir, cv::Mat image2);
	cv::Mat loadRegionOfInterest(std::string workingDir, std::string filename, cv::Mat image);

	int pow2roundup(int x);
};

Register::Register(std::string dir, std::string sampleName, std::string id, double lambda, int initialSlice, int numImages, std::string mumfordAppLocation, std::string roiSelectionApp, bool log) {
	this->dir = dir;
	this->sampleName = sampleName;
	this->lambda = lambda;
    this->NUM_IMAGES = numImages;
    this->INITIAL_SLICE = initialSlice;
    this->log = log;
	//strings
	ORIGINAL = "original.tiff";
	ROI_MASK_FILE = "roi_mask.tiff";
	CLEAN = "clean.tiff";
	MUMFORD = "mumford.bmp";
	ALIGNED = "aligned.tiff";
	TRANSLATE_SCALE_FILE = "translate_scale_info.txt";
	MUMFORD_APP_LOCATION = mumfordAppLocation;
	MUMFORD_LOG = "mumford.log";
	MUMFORD_ROTATED = "mumford_rotated.bmp";
	ROTATION_INFO_FILE = "rotation_info.txt";
	ROTATE_LOG_FILE = "rotation_scores.txt";
	COARSE_RESULT = "coarse.tiff";
	MASK_RESULT1 = "region_mask1.tiff";
	MASK_RESULT2 = "region_mask2.tiff";
	COARSE_AFFINE_MAT_FILE = "coarse_affine_matrix.txt";
	MASK = "sift_mask.tiff";
	REGISTERED = "registered.tiff";
	//transformation result of all iterations
	SIFT_AFFINE_MAT_FILE = "sift_affine_matrix.txt";

	SIFT_AFFINE_MAT_COST_FILE = "sift_affine_matrix_cost.txt";
	//logs each transformation result per iteration
	SIFT_AFFINE_MAT_LOG_FILE = "sift_affine_matrix_log.txt";
	//combines coarse and sift affine matrix
	FINAL_AFFINE_MAT_FILE = "final_affine_matrix.txt";
	//final affine matrixes for each slice
	FINAL_AFFINE_MAT_ALL_SLICES_FILE = "final_affine_matrix_all_slices_" + id + ".txt";

	ROI_LOCATION_FILE = "roi_location.txt";
	ROI_SELECTION_APP = roiSelectionApp;

	GRAY_COMMAND_1A = " 100 1 1 1223 100 0 >>  ";
	GRAY_COMMAND_1B = " 250 1 1 1223 100 0 >> ";
	GRAY_COMMAND_2A = " 300 1 1 1223 100 0 >> ";
	GRAY_COMMAND_A1 = " 100 1 1 1223 100 0 >> ";
	GRAY_COMMAND_A2 = " 100 1 1 1223 100 0 >> ";
	GRAY_COMMAND_A3 = " 300 1 1 1223 100 0 >> ";
	GRAY_COMMAND_B1 = " 100 1 1 1223 1000 0 >> ";

	workingDirSlice1 = dir + "/slice1/";

	std::cout << "Sample name: "<< sampleName <<"\t lambda: "<<lambda<< "\t ID: " <<id<<std::endl;

	start();


}

int Register::pow2roundup (int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

void Register::start() {

	for(int i=INITIAL_SLICE ;i <= INITIAL_SLICE+NUM_IMAGES-1;i++) { // different initial slice
		startSingle(i);
	}

	//saveAllSlicesFinalAffineMatrix();
}


void Register::startSingle(int index) {
	std::string workingDir = dir + "/slice" + std::to_string(index) + "/";

	std::cout<<"Slice "<<index<<std::endl;

//	WITH SCALING
	startCoarse(index, workingDir);
    startFine(index, workingDir);

	//saveFinalAffineMatrix(workingDir);
}

void Register::startCoarse(int index, std::string workingDir) {
	std::cout<<"Performing coarse registration (with scaling) ... ";

	doRoiInfo(index, workingDir);
	prepareForMumford(workingDir);
	doMumfordSegmentation(workingDir);
	std::string maskName1 = "mask" + std::to_string(index-1) + "_" + std::to_string(index) + ".tiff";
	std::string maskName2 = "mask" + std::to_string(index) + "_" + std::to_string(index+1) + ".tiff";
	if(index==INITIAL_SLICE) { // different initial slice
		//copy
		std::string filename1 = MUMFORD.substr(0, MUMFORD.find(".")) + "_out.bmp";
		std::string filename2 = MUMFORD.substr(0, MUMFORD.find(".")) + "_rotated.bmp";
		cv::Mat im = cv::imread(workingDir + filename1);
		cv::imwrite(workingDir + filename2, im);
	} else {
		std::string previousWorkingDir = dir + "/slice" + std::to_string(index-1) + "/";
		doRotation(previousWorkingDir, workingDir, maskName1);
	}

	transformImageForSIFT(workingDir, maskName1, maskName2, index);
}

void Register::startFine(int index, std::string workingDir) {
	std::cout<<"Performing fine registration (with scaling) ... ";

	if(index==INITIAL_SLICE) { // different initial slice
		cv::Mat eyeMat = cv::Mat();
		eyeMat = cv::Mat::eye(3, 2, CV_64FC1);
		Utils::saveAffineMatrixToFile(eyeMat,workingDir + SIFT_AFFINE_MAT_FILE);
		cv::Mat slice = cv::imread(workingDir + COARSE_RESULT);
		cv::imwrite(workingDir + REGISTERED, slice);
		cv::Mat mask = cv::imread(workingDir + ROI_MASK_FILE);
		cv::imwrite(workingDir + MASK, mask);
		cv::Mat image = cv::imread(workingDir + ORIGINAL);
		REFERENCE_IMAGE_WIDTH = pow2roundup(image.cols);
		REFERENCE_IMAGE_HEIGHT = pow2roundup(image.rows);
	    //setRegionOfInterest(workingDir,slice);

	} else {
		std::string previousWorkingDir = dir + "/slice" + std::to_string(index-1) + "/";
		std::string maskName = "mask" + std::to_string(index-1) + "_" + std::to_string(index) + ".tiff";
		performSIFTWithMask(previousWorkingDir, workingDir, maskName);
	}
}

void Register::doRoiInfo(int index, std::string workingDir) {
	std::cout<<"Getting roi info ... "<<std::endl;
	cv::Mat image = cv::imread(workingDir + ORIGINAL);
	cv::Mat padded_image;
	if(index == INITIAL_SLICE){ // different initial slice
		REFERENCE_IMAGE_WIDTH = pow2roundup(image.cols);
		REFERENCE_IMAGE_HEIGHT = pow2roundup(image.rows);
		int pad_left = (REFERENCE_IMAGE_WIDTH - image.cols)/2;
		int pad_right = (REFERENCE_IMAGE_WIDTH - image.cols) - pad_left;
		int pad_top = (REFERENCE_IMAGE_HEIGHT - image.rows)/2;
		int pad_bot = (REFERENCE_IMAGE_HEIGHT - image.rows) - pad_top;
		int borderType = BORDER_CONSTANT;
		copyMakeBorder(image, padded_image, pad_top, pad_bot, pad_left, pad_right, borderType, image.at<cv::Vec3b>(image.rows-1, image.cols-1));
		imwrite("original.tiff", padded_image);
	}
	else{
		REFERENCE_IMAGE_WIDTH = pow2roundup(image.cols);
		REFERENCE_IMAGE_HEIGHT = pow2roundup(image.rows);
		int pad_left = (REFERENCE_IMAGE_WIDTH - image.cols)/2;
		int pad_right = (REFERENCE_IMAGE_WIDTH - image.cols) - pad_left;
		int pad_top = (REFERENCE_IMAGE_HEIGHT - image.rows)/2;
		int pad_bot = (REFERENCE_IMAGE_HEIGHT - image.rows) - pad_top;
		int borderType = BORDER_CONSTANT;
		copyMakeBorder(image, padded_image, pad_top, pad_bot, pad_left, pad_right, borderType, image.at<cv::Vec3b>(image.rows-1, image.cols-1));
	}
	RoiInfo c(padded_image, sampleName, log);
	c.saveImages(workingDir);
	c.saveRoiMaskImage(workingDir + ROI_MASK_FILE);

}

void Register::doAlign(std::string workingDir) {
	std::cout<<"Aligning first slice ... "<<std::endl;
	cv::Point3i RoiInfo = RoiInfo::getRoiInformationFromFile(workingDir + ROI_INFO_FILE);
	cv::Mat image = cv::imread(workingDir + CLEAN);

	TranslateScale ts(RoiInfo, image);
	ts.saveGrayBMPImage(workingDir + MUMFORD);
	ts.saveTranslatedScaledImage(workingDir + ALIGNED);
	ts.saveTransformationInformation(workingDir + TRANSLATE_SCALE_FILE);

}

void Register::prepareForMumford(std::string workingDir) {
	std::cout<<"Aligning first slice ... "<<std::endl;
	cv::Mat image = cv::imread(workingDir + CLEAN);
	cv::Mat bmpImage;
	cv::resize(image, bmpImage, cv::Size(image.cols / 4, image.rows / 4));
	cvtColor(bmpImage, bmpImage, CV_BGR2GRAY);
	cv::imwrite(workingDir + MUMFORD, bmpImage);
	cv::imwrite(workingDir + ALIGNED, bmpImage);
}

void Register::prepareForMumford(std::string workingDir1, std::string workingDir2) {
	std::cout<<"Aligning slice to first slice ... "<<std::endl;
	cv::Mat image = cv::imread(workingDir2 + CLEAN);
	cv::Mat bmpImage;
	cv::resize(image, bmpImage, cv::Size(image.cols / 4, image.rows / 4));
	cvtColor(bmpImage, bmpImage, CV_BGR2GRAY);
	cv::imwrite(workingDir2 + MUMFORD, bmpImage);
	cv::imwrite(workingDir2 + ALIGNED, bmpImage);
}

void Register::doAlign(std::string workingDir1, std::string workingDir2) {
	std::cout<<"Aligning slice to first slice ... "<<std::endl;
	cv::Point3i RoiInfo1 = RoiInfo::getRoiInformationFromFile(workingDir1 + ROI_INFO_FILE);
	cv::Point3i RoiInfo2 = RoiInfo::getRoiInformationFromFile(workingDir2 + ROI_INFO_FILE);
	cv::Mat image = cv::imread(workingDir2 + CLEAN);

	TranslateScale ts(RoiInfo1, RoiInfo2, image);
	ts.saveGrayBMPImage(workingDir2 + MUMFORD);
	ts.saveTranslatedScaledImage(workingDir2 + ALIGNED);
	ts.saveTransformationInformation(workingDir2 + TRANSLATE_SCALE_FILE);
}

void Register::doMumfordSegmentation(std::string workingDir) {
	std::cout<<"Doing Mumford Shah Segmentation ... "<<std::endl;
	std::string filename = MUMFORD.substr(0, MUMFORD.find("."));

	std::string grayCommand;
	grayCommand = GRAY_COMMAND_1A;

	std::string command = "cd " + workingDir + "; " + MUMFORD_APP_LOCATION + " " +
				filename + grayCommand + MUMFORD_LOG;
	std::cout<<command<<std::endl;

	system(command.c_str());


}

void Register::doRotation(std::string workingDir1, std::string workingDir2, std::string maskName) {
	std::cout<<"Finding rotation parameters ... "<<std::endl;
	std::string filename1 = MUMFORD.substr(0, MUMFORD.find(".")) + "_rotated.bmp";
	std::string filename2 = MUMFORD.substr(0, MUMFORD.find(".")) + "_out.bmp";
	cv::Mat image1 = cv::imread(workingDir1 + filename1);
	cv::Mat image2 = cv::imread(workingDir2 + filename2);
	cv::Mat maskTemp1, maskTemp2;
	if(Utils::fileExist(workingDir1 + maskName)) {
		maskTemp1 = cv::imread(workingDir1 + maskName);
	}else{
		std::cout << "mask file is missing, generating one \n";
		maskTemp1 = Mat::zeros(image1.size(), image1.type());
	}
	if(Utils::fileExist(workingDir2 + maskName)) {
		maskTemp2 = cv::imread(workingDir2 + maskName);
	}else{
		std::cout << "mask file is missing, generating one \n";
		maskTemp2 = Mat::zeros(image1.size(), image1.type());
	}
	cv::Mat mask1, mask2;

	int pad_left = (REFERENCE_IMAGE_WIDTH - maskTemp1.cols)/2;
	int pad_right = (REFERENCE_IMAGE_WIDTH - maskTemp1.cols) - pad_left;
	int pad_top = (REFERENCE_IMAGE_HEIGHT - maskTemp1.rows)/2;
	int pad_bot = (REFERENCE_IMAGE_HEIGHT - maskTemp1.rows) - pad_top;
	int borderType = BORDER_CONSTANT;
	copyMakeBorder(maskTemp1, mask1, pad_top, pad_bot, pad_left, pad_right, borderType, 0);
	copyMakeBorder(maskTemp2, mask2, pad_top, pad_bot, pad_left, pad_right, borderType, 0);

	cv::Mat small_mask1, small_mask2;
	cv::resize(mask1, small_mask1, cv::Size(mask1.cols/8, mask1.rows/8));
	cv::resize(mask2, small_mask2, cv::Size(mask2.cols/8, mask2.rows/8));


	Rotate r(image1, image2, small_mask1, small_mask2);
	r.saveRotationInformation(workingDir2 + ROTATION_INFO_FILE);
	r.saveRotatedImage(workingDir2 + MUMFORD_ROTATED);
	if(log){
		r.saveScoreInformation(workingDir2 + ROTATE_LOG_FILE);
	}
}

void Register::transformImageForSIFT(std::string workingDir, std::string maskName1, std::string maskName2, int index) {
	std::cout<<"Preparing slice for SIFT ... "<<std::endl;


	cv::Mat finalMatrix;
	cv::Mat image = cv::imread(workingDir + CLEAN);


	if(Utils::fileExist(workingDir + ROTATION_INFO_FILE)) {

		cv::Point2i translate2 = Rotate::getTranslationFromFile(workingDir + ROTATION_INFO_FILE);
		int angle = Rotate::getAngleFromFile(workingDir + ROTATION_INFO_FILE);

		//CAUTION, REMEMBER THIS. times 4 due to downsized mumford shah image
		cv::Mat t2 = Utils::getTranslationMatrix(translate2.x * 4,translate2.y * 4);
		cv::Mat r = Utils::getRotationMatrix(angle, cv::Point(image.cols/2, image.rows/2));


		finalMatrix = r * t2;
	}
	else{
		finalMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	}
	cv::Mat affineMatrix = Utils::getAffineMatrixFrom3X3Matrix(finalMatrix);
	cv::Mat warpedImage;
	cv::warpAffine(image, warpedImage, affineMatrix, image.size());
	Utils::floodCorners(warpedImage, cv::Scalar (255.0, 255.0, 255.0),
			cv::Scalar (0.0, 0.0, 0.0), cv::Scalar (254.0, 254.0, 254.0));
	cv::imwrite(workingDir + COARSE_RESULT, warpedImage);

	if (index > INITIAL_SLICE){ // different initial slice
		cv::Mat warpedMask1, warpedMask2;
		cv::Mat mask1, mask2;
		if(Utils::fileExist(workingDir + maskName1)) {
			mask1 = cv::imread(workingDir + maskName1);
		}else{
			std::cout << "Mask file is missing, generating the default one. \n";
			mask1 = Mat::zeros(image.size(), image.type());

		}
		if(Utils::fileExist(workingDir + maskName2)) {
			mask2 = cv::imread(workingDir + maskName2);
		}else{
			std::cout << "Mask file is missing, generating the default one. \n";
			mask2 = Mat::zeros(image.size(), image.type());

			}
		int pad_left = (REFERENCE_IMAGE_WIDTH - mask1.cols)/2;
		int pad_right = (REFERENCE_IMAGE_WIDTH - mask1.cols) - pad_left;
		int pad_top = (REFERENCE_IMAGE_HEIGHT - mask1.rows)/2;
		int pad_bot = (REFERENCE_IMAGE_HEIGHT - mask1.rows) - pad_top;
		int borderType = BORDER_CONSTANT;
		copyMakeBorder(mask1, mask1, pad_top, pad_bot, pad_left, pad_right, borderType, 0);
		copyMakeBorder(mask2, mask2, pad_top, pad_bot, pad_left, pad_right, borderType, 0);
		cv::resize(mask1, mask1, cv::Size(image.cols, image.rows));
		cv::resize(mask2, mask2, cv::Size(image.cols, image.rows));
		cv::warpAffine(mask1, warpedMask1, affineMatrix, mask1.size());
		cv::warpAffine(mask2, warpedMask2, affineMatrix, mask2.size());
		cv::imwrite(workingDir + MASK_RESULT1, warpedMask1);
		cv::imwrite(workingDir + MASK_RESULT2, warpedMask2);
	}else{
		cv::Mat warpedMask1, warpedMask2;
		cv::Mat mask2;
		if(Utils::fileExist(workingDir + maskName1)) {
			mask2 = cv::imread(workingDir + maskName2);
		}else{
			std::cout << "Mask file is missing, generating the default one.";
			mask2 = Mat::zeros(image.size(), image.type());
		}
		int pad_left = (REFERENCE_IMAGE_WIDTH - mask2.cols)/2;
		int pad_right = (REFERENCE_IMAGE_WIDTH - mask2.cols) - pad_left;
		int pad_top = (REFERENCE_IMAGE_HEIGHT - mask2.rows)/2;
		int pad_bot = (REFERENCE_IMAGE_HEIGHT - mask2.rows) - pad_top;
		int borderType = BORDER_CONSTANT;
		copyMakeBorder(mask2, mask2, pad_top, pad_bot, pad_left, pad_right, borderType, 0);
		cv::resize(mask2, mask2, cv::Size(image.cols, image.rows));
		cv::imwrite(workingDir + MASK_RESULT2, mask2);
	}

	Utils::saveAffineMatrixToFile(affineMatrix, workingDir + COARSE_AFFINE_MAT_FILE);
}

void Register::performSIFT(std::string workingDir1, std::string workingDir2) {
	cv::Mat image1 = cv::imread(workingDir1 + REGISTERED);
	cv::Mat image2 = cv::imread(workingDir2 + COARSE_RESULT);

	SIFTRegistration s(image1, image2, workingDir1 + SIFT_AFFINE_MAT_FILE, workingDir2, workingDirSlice1, ROI_LOCATION_FILE, lambda, true, log);
	s.saveRegisteredImage(workingDir2 + REGISTERED);
	if(log){
		s.saveSIFTAffineMatrix(workingDir2 + SIFT_AFFINE_MAT_FILE,
				workingDir2 + SIFT_AFFINE_MAT_LOG_FILE);
		s.saveSIFTAffineMatrixCost(workingDir2 + SIFT_AFFINE_MAT_COST_FILE);
	}
}

void Register::saveFinalAffineMatrix(std::string workingDir) {
	std::string coarse = workingDir + COARSE_AFFINE_MAT_FILE;
	std::string sift = workingDir + SIFT_AFFINE_MAT_FILE;
	std::string final = workingDir + FINAL_AFFINE_MAT_FILE;

	cv::Mat cMat = Utils::loadAffineMatrixFromFile(coarse);
	cv::Mat sMat = Utils::loadAffineMatrixFromFile(sift);

	cv::Mat mult;
	mult = Utils::get3X3MatrixFromAffineMatrix(sMat) *
			Utils::get3X3MatrixFromAffineMatrix(cMat);
	cv::Mat finalMat = Utils::getAffineMatrixFrom3X3Matrix(mult);

	Utils::saveAffineMatrixToFile(finalMat, final);
}

void Register::saveAllSlicesFinalAffineMatrix() {
	std::vector<cv::Mat> mats;
	for(int i = 1; i <= INITIAL_SLICE+NUM_IMAGES-1; i++) { // different initial slice
		std::string matfile = dir + "/slice" + std::to_string(i) + "/" + FINAL_AFFINE_MAT_FILE;
		mats.push_back(Utils::loadAffineMatrixFromFile(matfile));
	}

	std::string outfile = dir + "/" + FINAL_AFFINE_MAT_ALL_SLICES_FILE;
	Utils::saveAffineMatrixToFile(mats, outfile);
}

/* ----------------------------------------------------------------------------------*/
/* ----------------------------------------------------------------------------------*/
/* -----------------------------NO SCALE STUFF---------------------------------------*/
/* ----------------------------------------------------------------------------------*/
/* ----------------------------------------------------------------------------------*/


void Register::setRegionOfInterest(std::string workingDir, cv::Mat image){
	std::string command = "python2 " + ROI_SELECTION_APP + " " + workingDir + "/" +COARSE_RESULT;
	std::cout<<command<<std::endl;
	system(command.c_str());
}



void Register::performSIFTWithMask(std::string workingDir1, std::string workingDir2, std::string maskName) {
	std::cout<<"Performing fine registration (with scaling and mask) ... "<<std::endl;

	cv::Mat image1 = cv::imread(workingDir1 + REGISTERED);
	cv::Mat image2 = cv::imread(workingDir2 + COARSE_RESULT);

	cv::Mat region_mask1 = cv::imread(workingDir1 + MASK_RESULT2);
	cv::Mat region_mask2 = cv::imread(workingDir2 + MASK_RESULT1);

	std::cout << region_mask1.size() << "\n";
	std::cout << image1.size() << "\n";


	cv::Mat mask1 = cv::imread(workingDir1 + MASK);
	cv::Mat mask2 = cv::imread(workingDir2 + ROI_MASK_FILE);

	cv::Mat scaleMat;
	scaleMat = cv::Mat::eye(3, 3, CV_64FC1);


	SIFTRegistration s(image1, image2, mask1, mask2, region_mask1, region_mask2, workingDir1 + SIFT_AFFINE_MAT_FILE, scaleMat, workingDir2,
			workingDirSlice1, ROI_LOCATION_FILE, lambda, true, log);
	s.saveRegisteredImage(workingDir2 + REGISTERED);
	s.saveMaskImage(workingDir2 + MASK);
	s.saveRegionMaskImage(workingDir2 + MASK_RESULT2);
	s.saveSIFTAffineMatrix(workingDir2 + SIFT_AFFINE_MAT_FILE,
			workingDir2 + SIFT_AFFINE_MAT_LOG_FILE);
	if(log){
		s.saveSIFTAffineMatrixCost(workingDir2 + SIFT_AFFINE_MAT_COST_FILE);
	}

}

#endif /* REGISTER_HPP_ */
