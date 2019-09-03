#ifndef SIFT_HPP_
#define SIFT_HPP_

#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv.h>
#include <set>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/plot.hpp>


typedef std::vector<cv::KeyPoint> KeyPoints;
typedef std::vector<cv::DMatch> Matches;


class SIFTRegistration {
public:

	SIFTRegistration(cv::Mat image1, cv::Mat image2,std::string siftInitMatFile, std::string saveDirectory, std::string firstSliceDirectory,
			std::string roiLocationFile, double lambda, bool permutate, bool log);

	SIFTRegistration(cv::Mat image1, cv::Mat image2, cv::Mat mask1, cv::Mat mask2, cv::Mat region_mask1, cv::Mat region_mask2,
				std::string siftInitMatFile, cv::Mat scaleMat, std::string saveDirectory, std::string firstSliceDirectory,
				std::string roiLocationFile, double lambda, bool permutate, bool log);

	void saveRegisteredImage(const std::string filename);
	void saveSIFTAffineMatrix(const std::string siftMatFile,
			const std::string siftMatLogFile);
	void saveSIFTAffineMatrixCost(const std::string siftMatCostFile);

	void saveMaskImage(const std::string filename);
	void saveRegionMaskImage(const std::string filename);

	double getImageDifferenceNormalizedMasked(const cv::Mat im1, const cv::Mat im2,
			const cv::Mat mask1, const cv::Mat mask2, cv::Mat region_mask1, cv::Mat region_mask2);

	double getNormalizedMutualInformationMasked(const cv::Mat im1,
			const cv::Mat im2, const cv::Mat mask1, const cv::Mat mask2, cv::Mat region_mask1, cv::Mat region_mask2);

	double getNormalizedMutualInformation(const cv::Mat im1,const cv::Mat im2);

	double getEntropy(double * probL, int numBins);

	double getNormalizedCrossCorrelationMasked(const cv::Mat im1,
			const cv::Mat im2, const cv::Mat mask1, const cv::Mat mask2);

	double getNormalizedCrossCorrelation(const cv::Mat im1,const cv::Mat im2);

	void getJointHistFromImages (cv::Mat im1, cv::Mat im2, cv::Mat cmask1, cv::Mat cmask2,  double  (&lCh)[256][256], int &numPixels);
	void getJointHistFromImages (cv::Mat im1, cv::Mat im2,  double  (&lCh)[256][256], int &numPixels);

	void getHistFromImage (cv::Mat img, cv::Mat msk,  double (&lCh)[256], int &numPixels);
	void getHistFromImage (cv::Mat img,  double (&lCh)[256], int &numPixels);
private:
	//slice 1 and slice 2 to be registered.
	//slices 2 will be transformed to match slice1
	cv::Mat image1, image2, mask1, mask2, region_mask1, region_mask2;
	cv::Mat resolution1, resolution2, lowres_mask1, lowres_mask2, lowres_regionmask1, lowres_regionmask2;

	cv::Mat roi;
	std::string roiLocationFile;

	//directory for save all log and intermediate results
	std::string saveDirectory, firstSliceDirectory;

	//file that contains the affine matrix to intialize slice2
	//usually found in slice 1 directory
	std::string siftInitMatFile;
	cv::Mat siftInitMat;

	//do threshold or cutoff
	//do permutate or use all at once
	bool threshold, permutate, log;

	//contains affine matrix for each sigma iteration
	std::vector<cv::Mat> siftMats;

	//scale
		cv::Mat scaleMat;

	//PARAMETERS
	unsigned int numMatches;
	int numSIFTIterations;
	float SIFTMatchRatio;
	//lamba
	double lambda;

	struct RegistrationInfo {
	public:
		Matches matches;
		cv::Mat affineMatrix;

		double imageScore, matrixScore, scoreToBeat;
		bool better;

		RegistrationInfo(Matches matches, cv::Mat affineMatrix,
				double imageScore, double matrixScore, double scoreToBeat, bool better);
	};

	struct AffineMatrixComparator {
		bool operator()(const cv::Mat &mat1, const cv::Mat &mat2) const {
			if( mat1.at<double>(0,0) < mat2.at<double>(0,0) ) { return true; }
			else if( mat1.at<double>(0,0) > mat2.at<double>(0,0) ) { return false;}
			else if( mat1.at<double>(0,1) < mat2.at<double>(0,1) ) { return true; }
			else if( mat1.at<double>(0,1) > mat2.at<double>(0,1) ) { return false;}
			else if( mat1.at<double>(0,2) < mat2.at<double>(0,2) ) { return true; }
			else if( mat1.at<double>(0,2) > mat2.at<double>(0,2) ) { return false;}
			else if( mat1.at<double>(1,0) < mat2.at<double>(1,0) ) { return true; }
			else if( mat1.at<double>(1,0) > mat2.at<double>(1,0) ) { return false;}
			else if( mat1.at<double>(1,1) < mat2.at<double>(1,1) ) { return true; }
			else if( mat1.at<double>(1,1) > mat2.at<double>(1,1) ) { return false;}
			else if( mat1.at<double>(1,2) < mat2.at<double>(1,2) ) { return true; }
			else if( mat1.at<double>(1,2) > mat2.at<double>(1,2) ) { return false;}
			else { return false; }
		}
	};

	void initialRegistration(cv::Mat image);
	void initialRegistration(cv::Mat image, cv::Mat mask);

	void saveScaledImage(const std::string filename, cv::Mat res);

	void transformMask(const cv::Mat inputImage, const cv::Mat affineMat, cv::Mat &outputImage);

	cv::Mat loadInitialAffineMatrix();
	void loadRegionOfInterest();

	cv::Mat doSIFT(int iteration);
	cv::Mat doSIFTTemp(int iteration);
	cv::Rect cropBoundingBox (cv::Mat res1, cv::Mat res2, int interation, cv::Mat &cropped_res1, cv::Mat &cropped_res2);

	void initializeParameters();
	void doubleMatFromVec3b(cv::Vec3b in, double (&arr) [3]);

	//----------- LOGGING ---------------//
	void logKeyPointsAndDescriptors(const KeyPoints kp1, const KeyPoints kp2,
			const cv::Mat des1, const cv::Mat des2, const cv::Mat res1,
			const cv::Mat res2, const int level);
	void logMatches(const Matches good_matches,
			const Matches matches_outliers_removed,
			const Matches selected_matches, const KeyPoints kp1,
			const KeyPoints kp2, const int level);
	void logChosenMatches(const Matches selected_matches,
			const KeyPoints kp1, const KeyPoints kp2, const int level);
	void logCropping(cv::Mat image1, cv::Mat image2,
			cv::Mat mask1, cv::Mat mask2, cv::Rect roi, int level);
	void logTransformation(const KeyPoints kp1, const KeyPoints kp2,
			const std::vector<RegistrationInfo> regInfo, const int level);
	void logRegisteredImageAndAffineMatrix(const cv::Mat affineMat, const int level);
	//-----------------------------------//

	cv::Mat multiplyTransformations(cv::Mat affineMat1, cv::Mat affineMat2);
	void addCoordinatesToDescriptors(KeyPoints kp, cv::Mat &des, int level);
	void addScaledCoordinatesToDescriptors(KeyPoints kp, cv::Mat &des, cv::Mat scaleMat, int level);
	void getGoodMatches(Matches &good_matches, cv::Mat des1, cv::Mat des2);

	void removeUniqueMatches(KeyPoints kp1, KeyPoints kp2,
			Matches &good_matches);

	void removeBadFeatureDistance(KeyPoints kp1, KeyPoints kp2,
			Matches &good_matches, int distance);

	void removeBadEuclideanDistance(KeyPoints kp1, KeyPoints kp2,
			Matches &good_matches, int distance);

	void rankAndRemoveFeature(KeyPoints kp1, KeyPoints kp2,
			Matches &good_matches, int numPt);

	void rankAndRemoveEuclidean(KeyPoints kp1, KeyPoints kp2,
			Matches &good_matches, int numPt);

	double getImageDifference(const cv::Mat im1, const cv::Mat im2);
	double getImageDifferenceNormalized(const cv::Mat im1, const cv::Mat im2);
	double getImageDifference(const cv::Mat im1, const cv::Mat im2,
			const cv::Mat affineMatrix);
	double getImageDifference(const cv::Mat im1, const cv::Mat im2,
			const cv::Mat affineMatrix, const double identityScore);
	double getFrobeniusNorm(cv::Mat mat);


	void combination(int offset, int K, Matches &good_matches,
			std::vector<Matches> &allmatches, Matches &currentMatch);

	cv::Mat performRigidTransform(KeyPoints kp1, KeyPoints kp2,
			Matches good_matches, cv::Mat inputImage, cv::Mat &outputImage);

	void estimateAffineMatrix(const KeyPoints kp1, const KeyPoints kp2,
			const Matches good_matches, cv::Mat &affineMat);
	void estimateAffineMatrixInLowResImage(const KeyPoints kp1, const KeyPoints kp2,
			const Matches good_matches, cv::Mat &affineMat, cv::Rect roi);
	void estimateAffineMatrixInLowResImageScaled(const KeyPoints kp1,
			const KeyPoints kp2, const Matches good_matches, cv::Mat &affineMat, cv::Rect roi, cv::Mat scale);
	void transformImage(const cv::Mat inputImage, const cv::Mat affineMat,
			cv::Mat &outputImage);
	double getImageDifference2(const cv::Mat image1, const cv::Mat image2);
	double getAffineMatrixCost(const cv::Mat affineMatrix);
	bool isIdentityMatrix(const cv::Mat affineMatrix);

};


void SIFTRegistration::loadRegionOfInterest(){
	this->roi = Utils::loadROIFromFile(firstSliceDirectory + roiLocationFile);

}

cv::Rect SIFTRegistration::cropBoundingBox(cv::Mat res1, cv::Mat res2, int iteration, cv::Mat &cropped_res1, cv::Mat &cropped_res2){
	cv::Size size = res1.size();
	int current_x = roi.at<double>(0,0)/ pow(2, iteration);
	int current_y = roi.at<double>(0,1) / pow(2, iteration);

	if(current_x < 1 || current_y < 1){
		std::cout << "image resolution is very low. Center coordinates are on image borders.\n ";
		std::exit(1);
	}

	int min_x = current_x - roi.at<double>(1,0)/2;
	int min_y = current_y - roi.at<double>(1,1)/2;
	int max_x = current_x + roi.at<double>(1,0)/2;
	int max_y = current_y + roi.at<double>(1,1)/2;
	int minLength_x = std::numeric_limits<int>::max();
	int minLength_y = std::numeric_limits<int>::max();

	if (min_x < 0 || max_x > size.height-1){
		if(minLength_x > current_x)
			minLength_x = current_x;
		if (minLength_x > abs(size.height - current_x))
			minLength_x = abs(size.height - current_x);
	}
	else{
		minLength_x = roi.at<double>(1,0)/2;
	}
	if (min_y < 0 || max_y > size.width-1){
		if (minLength_y > current_y)
			minLength_y = current_y;
		if (minLength_y > abs(size.width - current_y))
			minLength_y = abs(size.width - current_y);
	}
	else{
		minLength_y = roi.at<double>(1,1)/2;
	}

	cv::Rect myROI(current_y - minLength_y, current_x - minLength_x, 2*minLength_y, 2*minLength_x);
	cropped_res1 = res1(myROI);
	cropped_res2 = res2(myROI);
	return myROI;

}

SIFTRegistration::SIFTRegistration(cv::Mat image1, cv::Mat image2,
		std::string siftInitMatFile, std::string saveDirectory, std::string firstSliceDirectory, std::string roiLocationFile,
		double lambda=0, bool permutate = true, bool log = true) {
	image1.copyTo(this->image1);
	image2.copyTo(this->image2);
	this->siftInitMatFile = siftInitMatFile;
	this->saveDirectory = saveDirectory;
	this->firstSliceDirectory = firstSliceDirectory;
	this->permutate = permutate;
	this->scaleMat = scaleMat;
	this->lambda = lambda;
	this->log = log;

	initializeParameters();
	loadRegionOfInterest();

	double rx = roi.at<double>(1,0);
	double ry = roi.at<double>(1,1);


	std::cout<<"lambda: "<<lambda<<
			"\nSIFT iterations: "<< numSIFTIterations <<
			"\nNumber of matches: "<< numMatches <<
			"\nSIFT match ratio: "<< SIFTMatchRatio <<std::endl;


	initialRegistration(this->image2);


	for (int i = numSIFTIterations - 1; i >= 0; i--) {
		cv::Mat currentMat = doSIFT(i);
		siftMats.push_back(currentMat);
	}

}

SIFTRegistration::SIFTRegistration(cv::Mat image1, cv::Mat image2, cv::Mat mask1, cv::Mat mask2, cv::Mat region_mask1, cv::Mat region_mask2,
		std::string siftInitMatFile, cv::Mat scaleMat, std::string saveDirectory, std::string firstSliceDirectory,
		std::string roiLocationFile, double lambda=0, bool permutate = true, bool log= true){
	image1.copyTo(this->image1);
	image2.copyTo(this->image2);
	mask1.copyTo(this->mask1);
	mask2.copyTo(this->mask2);
	region_mask1.copyTo(this->region_mask1);
	region_mask2.copyTo(this->region_mask2);
	this->siftInitMatFile = siftInitMatFile;
	this->saveDirectory = saveDirectory;
	this->firstSliceDirectory = firstSliceDirectory;
	this->roiLocationFile = roiLocationFile;
	this->permutate = permutate;
	this->scaleMat = scaleMat;
	this->lambda = lambda;
	this->log = log;
	initializeParameters();
	loadRegionOfInterest();

	double rx = roi.at<double>(1,0);
	double ry = roi.at<double>(1,1);

	std::cout << "lambda: " << lambda << "\tSIFT iterations: "
			<< numSIFTIterations << "\tNumber of matches: " << numMatches
			<< "\tSIFT match ratio: " << SIFTMatchRatio << std::endl;

	initialRegistration(this->image2, this->mask2);
	if(log){
		saveScaledImage("coarse-after-prev-slide-sift-application", this->image2);
	}
	for (int i = numSIFTIterations - 1; i >= 0; i--) {
		cv::Mat currentMat = doSIFTTemp(i);
		siftMats.push_back(currentMat);
	}

}

void SIFTRegistration::saveMaskImage(const std::string filename) {
	imwrite(filename, mask2);
}

void SIFTRegistration::saveRegionMaskImage(const std::string filename) {
	imwrite(filename, region_mask2);
}

double SIFTRegistration::getImageDifferenceNormalizedMasked(const cv::Mat im1,
		const cv::Mat im2, const cv::Mat mask1, const cv::Mat mask2, cv::Mat region_mask1, cv::Mat region_mask2) {

	int counter = 0;
	double sum = 0;

	for (int i = 0; i < im1.rows; i++) {
		for (int j = 0; j < im1.cols; j++) {
			cv::Vec3b v1 = im1.at<cv::Vec3b>(i, j);
			cv::Vec3b v2 = im2.at<cv::Vec3b>(i, j);

			cv::Vec3b m1 = mask1.at<cv::Vec3b>(i, j);
			cv::Vec3b m2 = mask2.at<cv::Vec3b>(i, j);
			cv::Vec3b rm1 = region_mask1.at<cv::Vec3b>(i, j);
			cv::Vec3b rm2 = region_mask2.at<cv::Vec3b>(i, j);
			//if all black ignore
			if(m1[0] == 255 && rm1[0] == 0	&& m2[0] == 255
					&& rm2[0] == 0){
					//combined_mask1.at<uchar>(i, j) = 255;
				double temp1 = (v1[0] - v2[0]) * (v1[0] - v2[0]);
				sum += sqrt(temp1);
				counter++;
			}
		}
	}
	double SSD = 0;
	if(counter!=0)
		SSD = sum/(double)counter;
	else
		SSD = std::numeric_limits<double>::infinity();

	return SSD;
}


double SIFTRegistration::getEntropy(double * probL, int numBins){
	double entropy = 0;
	for(int i = 0; i < numBins; i++){
		if(probL[i]!=0){
			entropy -= probL[i] * std::log(probL[i]);
		}
	}
	return entropy;
}


void SIFTRegistration::getHistFromImage(cv::Mat img, double (&lCh)[256], int &numPixels) {

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
				cv::Vec3b v = img.at<cv::Vec3b>(i, j);
				lCh[v[0]]++;
				numPixels++;
		}
	}
}

void SIFTRegistration::getHistFromImage (cv::Mat img, cv::Mat msk, double (&lCh)[256], int &numPixels) {


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
				cv::Vec3b v = img.at<cv::Vec3b>(i, j);
				cv::Vec3b m = msk.at<cv::Vec3b>(i, j);
				if(m[0] == 255) {
					lCh[v[0]]++;
					numPixels++;
				}
			}
		}
}

void SIFTRegistration::getJointHistFromImages (cv::Mat im1, cv::Mat im2, double  (&lCh)[256][256], int &numPixels) {


	for (int i = 0; i < im1.rows; i++) {
		for (int j = 0; j < im1.cols; j++) {

				cv::Vec3b v1 = im1.at<cv::Vec3b>(i, j);
				cv::Vec3b v2 = im2.at<cv::Vec3b>(i, j);
				lCh[v1[0]][v2[0]]++;
				numPixels++;

			}
		}
}

void SIFTRegistration::getJointHistFromImages (cv::Mat im1, cv::Mat im2, cv::Mat cmask1, cv::Mat cmask2, double  (&lCh)[256][256], int &numPixels) {


	for (int i = 0; i < im1.rows; i++) {
		for (int j = 0; j < im1.cols; j++) {

				cv::Vec3b v1 = im1.at<cv::Vec3b>(i, j);
				cv::Vec3b v2 = im2.at<cv::Vec3b>(i, j);
				cv::Vec3b m1 = cmask1.at<cv::Vec3b>(i, j);
				cv::Vec3b m2 = cmask2.at<cv::Vec3b>(i, j);

				//if black ignore
				if(m1[0] == 255 && m2[0] == 255) {
					lCh[v1[0]][v2[0]]++;
					numPixels++;

				}
			}
		}
}

double SIFTRegistration::getNormalizedMutualInformation(const cv::Mat im1,const cv::Mat im2){

	int numPixels1 = 0;
	int numPixels2 = 0;
	int numPixels = 0;
	double lCh1[256];
	for (int i = 0; i < 256; i++)
			lCh1[i] = 0;
	double lCh2[256];
	for (int i = 0; i < 256; i++)
			lCh2[i] = 0;
	double lCh[256][256];
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++)
			lCh[i][j] = 0;
	getHistFromImage(im1, lCh1, numPixels1);
	getHistFromImage(im2, lCh2, numPixels2);
	getJointHistFromImages(im1, im2, lCh, numPixels);
	double probL1[256];
	double probL2[256];
	double probL[256][256];
	if ( numPixels1!= 0){
		for (int i = 0; i < 256; i++) {
			probL1[i] = lCh1[i]/(double)numPixels1;
		}
	}

	if ( numPixels2!= 0){
		for (int i = 0; i < 256; i++) {
			probL2[i] = lCh2[i]/(double)numPixels2;
			}
	}
    if (numPixels !=0){
    	for (int i = 0; i < 256; i++) {
    		for(int j=0; j < 256; j++){
    			probL[i][j] = lCh[i][j]/(double)numPixels;
    		}
    	}
    }
    double MI = 0;
    for (int i = 0; i < 256; i++) {
        for(int j = 0; j < 256; j++){
        	if (probL1[i]!=0 && probL2[i]!=0 && probL[i][j]!=0){
        		MI += probL[i][j]* std::log(probL[i][j]/(probL1[i]*probL2[j]));
        	}
        }
    }
    double Entropy1 = getEntropy(probL1, 256);
    double Entropy2 = getEntropy(probL2, 256);
    double NMI = (2* MI)/(Entropy1 + Entropy2);
	return -NMI;
}

double SIFTRegistration::getNormalizedMutualInformationMasked(const cv::Mat im1,const cv::Mat im2, const cv::Mat mask1, const cv::Mat mask2,
		cv::Mat region_mask1, cv::Mat region_mask2){

	cv::Mat regmask1 = cv::Mat();
	cv::Mat regmask2 = cv::Mat();
	cv::Mat cmask1 = cv::Mat();
	cv::Mat cmask2 = cv::Mat();
	cv::Mat cmask =  cv::Mat();
	regmask1 =  255 * cv::Mat::ones(region_mask1.size(), region_mask1.type()) - region_mask1;
	regmask2 = 255 * cv::Mat::ones(region_mask2.size(), region_mask2.type()) - region_mask2;
	cmask1 =  mask1.mul(regmask1);
	cmask2 = mask2.mul(regmask2);
	cmask =  cmask1.mul(cmask2);
	int numBins = 256;
	double lCh1[256] = {0};
	double lCh2[256] = {0};
	double lCh[256][256];
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++)
			lCh[i][j] = 0;
	int numPixels1 = 0;
	int numPixels2 = 0;
	int numPixels = 0;
	getHistFromImage(im1, cmask1, lCh1, numPixels1);
	getHistFromImage(im2, cmask2, lCh2, numPixels2);
	getJointHistFromImages(im1, im2, cmask1, cmask2, lCh, numPixels);
	double probL1[256];
	double probL2[256];
	double probL[256][256];
	if ( numPixels1!= 0){
		for (int i = 0; i < numBins; i++) {
			probL1[i] = lCh1[i]/(double)numPixels1;
		}
	}

	if ( numPixels2!= 0){
		for (int i = 0; i < numBins; i++) {
			probL2[i] = lCh2[i]/(double)numPixels2;
		}
	}
    if (numPixels !=0){
    	for (int i = 0; i < numBins; i++) {
    		for(int j=0; j < numBins; j++){
    			probL[i][j] = lCh[i][j]/(double)numPixels;
    		}
    	}
    }
    double MI = 0;
    for (int i = 0; i < numBins; i++) {
        for(int j = 0; j < numBins; j++){
        	if (probL1[i]!=0 && probL2[i]!=0 && probL[i][j]!=0){
        		MI += probL[i][j]* std::log(probL[i][j]/(probL1[i]*probL2[j]));
        	}
        }
    }
    double Entropy1 = getEntropy(probL1, numBins);
	double Entropy2 = getEntropy(probL2, numBins);
	double NMI = (2* MI)/(Entropy1 + Entropy2);
	return -NMI;
}


double SIFTRegistration::getNormalizedCrossCorrelationMasked(const cv::Mat im1,
		const cv::Mat im2, const cv::Mat mask1, const cv::Mat mask2) {
	cv::Mat gray1, gray2;
	cvtColor(im1, gray1, CV_BGR2GRAY);
	cvtColor(im2, gray2, CV_BGR2GRAY);
	int counter1 = 0, counter2 = 0;
	double v1mean = 0;
	double v2mean = 0;
	double numer1 = 0;
	double denom11 = 0;
	double denom12 = 0;

	for (int i = 0; i < mask1.rows; i++) {
		for (int j = 0; j < mask1.cols; j++) {
			cv::Vec3b m1;
			m1 = mask1.at<cv::Vec3b>(i, j);
			//if all black ignore
			if( m1[0] == 0 && m1[1]==0) {
				continue;
			} else {
				double v1 = gray1.at<uchar>(i, j);
				v1mean = v1mean + v1;
				counter1 = counter1 + 1;
			}
		}
	}

	for (int i = 0; i < mask2.rows; i++) {
		for (int j = 0; j < mask2.cols; j++) {
			cv::Vec3b m2 = mask2.at<cv::Vec3b>(i, j);
			//if all black ignore
			if( m2[0] == 0 && m2[1]==0) {
				continue;
			} else {
				double v2 = gray2.at<uchar>(i, j);
				v2mean = v2mean + v2;
				counter2 = counter2 + 1;
			}
		}
	}


	if(counter1!=0){
		v1mean= v1mean  / double(counter1);
	}
	if(counter2!=0){
		v2mean  = v2mean / double(counter2);
	}


	for (int i = 0; i < im1.rows; i++) {
			for (int j = 0; j < im1.cols; j++) {
				cv::Vec3b m1 = mask1.at<cv::Vec3b>(i, j);
				cv::Vec3b m2 = mask2.at<cv::Vec3b>(i, j);

				//if all black ignore
				if( m1[0] == 0 && m1[1]==0  && m2[0] == 0 && m2[1] == 0) {
					continue;
				} else {
					double v1 = gray1.at<uchar>(i, j);
					double v2 = gray2.at<uchar>(i, j);
					//get pixel squared difference and add to counter

					double diff1;
					double diff2;
					diff1 = v1 - v1mean;
					diff2 = v2 - v2mean;
					numer1 = numer1 +  (diff1)*(diff2) ;
					denom11 = denom11 + pow(diff1,2);
					denom12 = denom12 + pow(diff2,2);
			}
		}
	}

	double NCC = 0;
	if(denom11!=0 && denom12 !=0 )
		NCC = (numer1/std::sqrt(denom11*denom12));

	return NCC;
}

void SIFTRegistration::doubleMatFromVec3b(cv::Vec3b in, double (&arr) [3])
{
    arr[0] = in [0];
    arr[1] = in [1];
    arr[2] = in [2];

}

double SIFTRegistration::getNormalizedCrossCorrelation(const cv::Mat im1,const cv::Mat im2) {

	cv::Mat gray1, gray2;
	cvtColor(im1, gray1, CV_BGR2GRAY);
	cvtColor(im2, gray2, CV_BGR2GRAY);
	int counter1 = 0, counter2 = 0;
	double v1mean = 0;
	double  v2mean = 0;
	double numer1 = 0;
	double denom11 = 0;
	double denom12 = 0;

	for (int i = 0; i < im1.rows; i++) {
		for (int j = 0; j < im1.cols; j++) {
			double v1 = gray1.at<uchar>(i, j);
			//if all black ignore
			if( v1 == 255 ) {
				continue;
			} else {
				v1mean = v1mean + v1;
				counter1 = counter1 + 1;
			}
		}
	}

	for (int i = 0; i < im2.rows; i++) {
		for (int j = 0; j < im2.cols; j++) {
			double v2 = gray2.at<uchar>(i, j);
			//if all black ignore
			if( v2 == 255 ) {
				continue;
			} else {
				v2mean= v2mean + v2;
				counter2 = counter2 + 1;
			}
		}
	}

	if(counter1!=0){
		v1mean = v1mean  / double(counter1);
	}
	if(counter2!=0){
		v2mean = v2mean  / double(counter2);
	}

	for (int i = 0; i < im1.rows; i++) {
			for (int j = 0; j < im1.cols; j++) {
				double v1 = gray1.at<uchar>(i, j);
				double v2 = gray2.at<uchar>(i, j);
				//if all black ignore
				if( v1 == 255 && v2 == 255) {
					continue;
				} else {
				//get pixel squared difference and add to counter
					double diff1;
					double diff2 ;
					diff1 = v1 - v1mean;
					diff2 = v2 - v2mean;
					numer1 = numer1 +  (diff1)*(diff2) ;
					denom11 = denom11 + pow(diff1,2);
					denom12 = denom12 + pow(diff2,2);
			}
		}
	}

	double NCC = 0;
	if(denom11!=0 && denom12 !=0)
		NCC = numer1/std::sqrt(denom11*denom12);

	return NCC;
}
void SIFTRegistration::initialRegistration(cv::Mat image) {
	siftInitMat = loadInitialAffineMatrix();
	warpAffine(image2, image2, siftInitMat, image2.size(), cv::INTER_LANCZOS4+CV_WARP_FILL_OUTLIERS);

	cv::Scalar fill = cv::Scalar(255.0, 255.0, 255.0);
	cv::Scalar loDiff = cv::Scalar(0.0, 0.0, 0.0);
	cv::Scalar upDiff = cv::Scalar(254.0, 254.0, 254.0);
	Utils::floodCorners(image2, fill, loDiff, upDiff);
}

void SIFTRegistration::initialRegistration(cv::Mat image, cv::Mat mask) {
	siftInitMat = loadInitialAffineMatrix();
	warpAffine(image2, image2, siftInitMat, image2.size(), cv::INTER_LANCZOS4+CV_WARP_FILL_OUTLIERS);

	cv::Scalar fill = cv::Scalar(255.0, 255.0, 255.0);
	cv::Scalar loDiff = cv::Scalar(0.0, 0.0, 0.0);
	cv::Scalar upDiff = cv::Scalar(254.0, 254.0, 254.0);
	Utils::floodCorners(image2, fill, loDiff, upDiff);

	warpAffine(mask2, mask2, siftInitMat, mask2.size());
	Utils::floodCorners(mask2, cv::Scalar(0.0, 0.0, 0.0));
}

cv::Mat SIFTRegistration::multiplyTransformations(cv::Mat affineMat1, cv::Mat affineMat2){
	cv::Mat affine1 = cv::Mat();
	affine1 = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat affine2 =  cv::Mat();
	affine2 = cv::Mat::eye(3, 3, CV_64FC1);
	affineMat1.row(0).copyTo(affine1.row(0));
	affineMat1.row(1).copyTo(affine1.row(1));
    affineMat2.row(1).copyTo(affine2.row(1));
	affineMat2.row(0).copyTo(affine2.row(0));
	cv::Mat mul;
	try{
			mul = affine1*affine2;
	}
	catch (Exception& e){
		const char* err_msg = e.what();
		std::cout << err_msg;
	}
	cv::Mat result = cv::Mat();
	result = cv::Mat::eye(2, 3, CV_64FC1);
	mul.row(0).copyTo(result.row(0));
	mul.row(1).copyTo(result.row(1));

	return result;
}

cv::Mat SIFTRegistration::doSIFTTemp(int iteration) {
	KeyPoints kp1, kp2;
	cv::Mat des1, des2;
	cv::Mat orgResolution1, orgResolution2;
	cv::Mat orgMask1, orgMask2;
	cv::Mat orgRegionMask1, orgRegionMask2;
	Matches good_matches, matches_outliers_removed, selected_matches;
	double factor = (1.0/pow(2,iteration));

	std::cout << "Factor " << factor ; // << std::endl;

	//decrease resolution
	cv::resize(image1, orgResolution1, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);
	cv::resize(image2, orgResolution2, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);
	cv::resize(mask1, orgMask1, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);
	cv::resize(mask2, orgMask2, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);
	cv::resize(region_mask1, orgRegionMask1, cv::Size(0, 0), factor, factor,cv::INTER_LANCZOS4);
	cv::resize(region_mask2, orgRegionMask2, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);

	cropBoundingBox(orgResolution1, orgResolution2, iteration, resolution1, resolution2);
	cv::Rect roi = cropBoundingBox(orgMask1, orgMask2, iteration, lowres_mask1, lowres_mask2);
	logCropping(orgResolution1, orgResolution2, orgMask1, orgMask2, roi, iteration);
	if(log){
		saveScaledImage("cropped_coarse_"+std::to_string(iteration), resolution2);
		saveScaledImage("cropped_mask_"+std::to_string(iteration), lowres_mask2);
		saveScaledImage("scaled_coarse_"+std::to_string(iteration), orgResolution2);
		saveScaledImage("scaled_mask_"+std::to_string(iteration), orgMask2);
	}


	cv::Ptr<cv::FeatureDetector> detector;
	detector = cv::xfeatures2d::SIFT::create(0, 10);//0, 10
	detector->detect(resolution1, kp1);
	detector->detect(resolution2, kp2);
	cv::Ptr<cv::DescriptorExtractor> extractor;
	extractor = cv::xfeatures2d::SIFT::create(0, 10);// 0, 10
	extractor->compute(resolution1, kp1, des1);
	extractor->compute(resolution2, kp2, des2);
    if(log){
    	logKeyPointsAndDescriptors(kp1, kp2, des1, des2, resolution1, resolution2, iteration);
    }
	if (kp1.size() <= 1 || kp2.size() <= 1) {
		std::cout << "Not enough keypoints" << std::endl;
		return cv::Mat::eye(2, 3, CV_64FC1);
	}
    cv::Mat affineMat = cv::Mat();
	affineMat = cv::Mat::eye(2, 3, CV_64FC1);
	std::vector<RegistrationInfo> regInfo;

	//added to check if measuring the distance for the local roi region only works better
	cv::Mat tempResolution1, tempResolution2;
	cv::Mat tempMask1, tempMask2;
	cv::Mat tempRegionMask1, tempRegionMask2;
	cropBoundingBox(orgResolution1, orgResolution2, iteration, tempResolution1, tempResolution2);
	cropBoundingBox(orgRegionMask1, orgRegionMask2, iteration, tempRegionMask1, tempRegionMask2);
	cropBoundingBox(orgMask1, orgMask2, iteration, tempMask1, tempMask2);

	addCoordinatesToDescriptors(kp1, des1, iteration);
	addScaledCoordinatesToDescriptors(kp2, des2, scaleMat, iteration);
	//Get Good Matches
	getGoodMatches(good_matches, des1, des2);

	//Remove outliers
	matches_outliers_removed = good_matches;
	removeUniqueMatches(kp1, kp2, matches_outliers_removed);

	//take top 8
	selected_matches = matches_outliers_removed;
	if (selected_matches.size() > numMatches) {
		selected_matches.erase(selected_matches.begin() + numMatches,
				selected_matches.end());
	}
	logMatches(good_matches, matches_outliers_removed, selected_matches, kp1,kp2, iteration);

	if (selected_matches.size() < 3) {
		std::cout << "Not enough matches" << std::endl;
		cv::Mat affineMat;
		affineMat = cv::Mat::eye(2, 3, CV_64FC1);
		if (log){
			logRegisteredImageAndAffineMatrix(affineMat, iteration);
		}
		return affineMat;
	}


	double identityImageScore = getImageDifferenceNormalized(tempResolution1, tempResolution2);
	double identityMatrixScore = getAffineMatrixCost(affineMat);
	double minScoreSoFar = identityImageScore + identityMatrixScore;


	if (permutate) {
		std::vector<Matches> allMatches;
		std::set<cv::Mat, AffineMatrixComparator> candidates;
		Matches temp;
		combination(0, 3, selected_matches, allMatches, temp);

		std::cout << "\t Total matches:  " << selected_matches.size()
				<< " Combinations:  " << allMatches.size() << std::endl;

		int bestMatchIndex = -1;

		//Add XY coordinates to Descriptors


		for (unsigned int i = 0; i < allMatches.size(); i++) {

				Matches matches = allMatches.at(i);
				cv::Mat cAffineMat;
				double cImageScore = 0, cMatrixScore = 0;

				estimateAffineMatrixInLowResImage(kp1, kp2, matches, cAffineMat, roi);

				if (candidates.find(cAffineMat) == candidates.end()) {

					candidates.insert(cAffineMat);

					if (isIdentityMatrix(cAffineMat)) {
						cImageScore = identityImageScore;
						cMatrixScore = identityMatrixScore;
					} else {
						cv::Mat cTransformedImage, cMaskedImage, cRegionMaskedImage;
						transformImage(orgResolution2, cAffineMat , cTransformedImage);
						if(lowres_mask2.data) {
							transformMask(orgMask2, cAffineMat , cMaskedImage);
							transformMask(orgRegionMask2, cAffineMat, cRegionMaskedImage);
							cv::Mat cResolution1, cResolution2;
							cv::Mat cMask1, cMask2;
							cv::Mat cRegionMask1, cRegionMask2;
							cropBoundingBox(orgResolution1, cTransformedImage, iteration, cResolution1, cResolution2);
							cropBoundingBox(orgMask1, cMaskedImage, iteration, cMask1, cMask2);
							cropBoundingBox(orgRegionMask1, cRegionMaskedImage, iteration, cRegionMask1, cRegionMask2);
							cImageScore = getImageDifferenceNormalized(cResolution1, cResolution2);

						} else {
							cv::Mat cResolution1, cResolution2;
							cropBoundingBox(orgResolution1, cTransformedImage, iteration, cResolution1, cResolution2);
							cImageScore = getImageDifferenceNormalized(cResolution1, cResolution2);
							//cImageScore = getNormalizedMutualInformation(cResolution1, cResolution2);
						}
						cMatrixScore = getAffineMatrixCost(cAffineMat);
					}

					double cScore = cMatrixScore + cImageScore;

					if (cScore < minScoreSoFar) {
						regInfo.push_back(
								RegistrationInfo(matches, cAffineMat, cImageScore,
										cMatrixScore, minScoreSoFar, true));
						minScoreSoFar = cScore;
						bestMatchIndex = i;
					} else {
						regInfo.push_back(
								RegistrationInfo(matches, cAffineMat, cImageScore,
										cMatrixScore, minScoreSoFar, false));
					}

				} else {
					regInfo.push_back(
							RegistrationInfo(matches, cAffineMat, -1,
									-1, minScoreSoFar, false));
				}

		}

		//Assign transformation to image2 if better than current
		if (bestMatchIndex >= 0) {
			Matches bestMatches = allMatches.at(bestMatchIndex);
			if (log){
				logChosenMatches(bestMatches, kp1,kp2, iteration);
			}
			estimateAffineMatrixInLowResImage(kp1, kp2, bestMatches, affineMat, roi);
			//Mahsa:
			affineMat.at<double>(0,2)= pow(2,iteration) * affineMat.at<double>(0,2);
			affineMat.at<double>(1,2)= pow(2,iteration) * affineMat.at<double>(1,2);
			transformImage(image2, affineMat, image2);
			transformMask(mask2, affineMat, mask2);
			transformMask(region_mask2, affineMat, region_mask2);
		}
		std::cout << "iteration: " << iteration << " bestMatchIndex: " << bestMatchIndex  << std::endl;
	} else { //use all points

		cv::Mat transformedImage, maskedImage, regionMaskedImage;


			estimateAffineMatrixInLowResImage(kp1, kp2, good_matches, affineMat, roi);
			double imageScore;
			transformImage(orgResolution2, affineMat, transformedImage);
			if(lowres_mask2.data) {
				transformMask(orgMask2, affineMat, maskedImage);
				transformMask(orgRegionMask2, affineMat, regionMaskedImage);
				cv::Mat cResolution1, cResolution2;
				cv::Mat cMask1, cMask2;
				cv::Mat cRegionMask1, cRegionMask2;
				cropBoundingBox(orgResolution1, transformedImage, iteration, cResolution1, cResolution2);
				cropBoundingBox(orgMask1, maskedImage, iteration, cMask1, cMask2);
				cropBoundingBox(orgRegionMask1, regionMaskedImage, iteration, cRegionMask1, cRegionMask2);
				imageScore = getImageDifferenceNormalized(cResolution1, cResolution2);
			} else {
				cv::Mat cResolution1, cResolution2;
				cropBoundingBox(orgResolution1, transformedImage, iteration, cResolution1, cResolution2);
				imageScore = getImageDifferenceNormalized(cResolution1, cResolution2);
			}

			double matrixScore = getAffineMatrixCost(affineMat);
			double cScore = matrixScore + imageScore;

			//Assign transformation to image2 if better than current
			if (cScore < minScoreSoFar) {
				estimateAffineMatrixInLowResImage(kp1, kp2, good_matches, affineMat, roi);
				affineMat.at<double>(0,2)= pow(2, iteration) *  affineMat.at<double>(0,2);
				affineMat.at<double>(1,2)= pow(2, iteration) * affineMat.at<double>(1,2);
				transformImage(image2, affineMat, image2);
				transformMask(mask2, affineMat, mask2);
				transformMask(region_mask2, affineMat, region_mask2);
				std::cout << "iteration: " << iteration << " using all points" << std::endl;
				regInfo.push_back(
						RegistrationInfo(good_matches, affineMat, imageScore,
								matrixScore, minScoreSoFar, true));

			} else {
				regInfo.push_back(
						RegistrationInfo(good_matches, affineMat, imageScore,
								matrixScore, minScoreSoFar, false));
			}
		}


		if(log){
			logRegisteredImageAndAffineMatrix(affineMat, iteration);
			logTransformation(kp1, kp2, regInfo, iteration);
		}

	return affineMat;

}
cv::Mat SIFTRegistration::doSIFT(int iteration) {

	KeyPoints kp1, kp2;
	cv::Mat des1, des2;
	cv::Mat orgResolution1, orgResolution2;
	cv::Mat orgMask1, orgMask2;
	cv::Mat orgRegionMask1, orgRegionMask2;
	Matches good_matches, matches_outliers_removed, selected_matches;
	double factor = (1.0/pow(2,iteration));

	std::cout << "Factor " << factor; // << std::endl;

	//decrease resolution
	cv::resize(image1, orgResolution1, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);
	cv::resize(image2, orgResolution2, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);
	cv::resize(mask1, orgMask1, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);
	cv::resize(mask2, orgMask2, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);
	cv::resize(region_mask1, orgRegionMask1, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);
	cv::resize(region_mask2, orgRegionMask2, cv::Size(0, 0), factor, factor, cv::INTER_LANCZOS4);


	cropBoundingBox(orgResolution1, orgResolution2, iteration, resolution1, resolution2);
	cv::Rect roi = cropBoundingBox(orgMask1, orgMask2, iteration, lowres_mask1, lowres_mask2);
	logCropping(orgResolution1, orgResolution2, orgMask1, orgMask2, roi, iteration);

	if(log){
			saveScaledImage("scaled_coarse_"+std::to_string(iteration), orgResolution2);
			saveScaledImage("scaled_mask_"+std::to_string(iteration), orgMask2);
			saveScaledImage("cropped_coarse_"+std::to_string(iteration), resolution2);
			saveScaledImage("cropped_mask_"+std::to_string(iteration), lowres_mask2);
	}

	cv::Ptr<cv::FeatureDetector> detector;
	detector = cv::xfeatures2d::SIFT::create(0, 10);//0, 10
	detector->detect(resolution1, kp1);
	detector->detect(resolution2, kp2);
	cv::Ptr<cv::DescriptorExtractor> extractor;
	extractor = cv::xfeatures2d::SIFT::create(0, 10);// 0, 10
	extractor->compute(resolution1, kp1, des1);
	extractor->compute(resolution2, kp2, des2);

	if(log){
		logKeyPointsAndDescriptors(kp1, kp2, des1, des2, resolution1, resolution2, iteration);
	}
	if (kp1.size() <= 1 || kp2.size() <= 1) {
		std::cout << "Not enough keypoints" << std::endl;
		return cv::Mat::eye(2, 3, CV_64FC1);
	}

	//Add XY coordinates to Descriptors
	addCoordinatesToDescriptors(kp1, des1, iteration);
	addScaledCoordinatesToDescriptors(kp2, des2, scaleMat, iteration);
	//Get Good Matches
	getGoodMatches(good_matches, des1, des2);

	//Remove outliers
	matches_outliers_removed = good_matches;
	removeUniqueMatches(kp1, kp2, matches_outliers_removed);

	//take top 8
	selected_matches = matches_outliers_removed;
	if (selected_matches.size() > numMatches) {
		selected_matches.erase(selected_matches.begin() + numMatches,
				selected_matches.end());
	}
	if(log){
		logMatches(good_matches, matches_outliers_removed, selected_matches, kp1,
			kp2, iteration);
	}
	if (selected_matches.size() < 3) {
		std::cout << "Not enough matches" << std::endl;
		cv::Mat affineMat;
		affineMat = cv::Mat::eye(2, 3, CV_64FC1);
		if(log){
			logRegisteredImageAndAffineMatrix(affineMat, iteration);
		}
		return affineMat;
	}

	//Find transformation
	cv::Mat affineMat;
	affineMat = cv::Mat::eye(2, 3, CV_64FC1);
	std::vector<RegistrationInfo> regInfo;

	double identityImageScore = getImageDifferenceNormalizedMasked(orgResolution1, orgResolution2, orgMask1, orgMask2, orgRegionMask1, orgRegionMask2);
	double identityMatrixScore = getAffineMatrixCost(affineMat);
	double minScoreSoFar = identityImageScore + identityMatrixScore;

	if (permutate) {
		std::vector<Matches> allMatches;
		std::set<cv::Mat, AffineMatrixComparator> candidates;
		Matches temp;
		combination(0, 3, selected_matches, allMatches, temp);

		std::cout << "\t Total matches:  " << selected_matches.size()
				<< " Combinations:  " << allMatches.size() << std::endl;

		int bestMatchIndex = -1;

		for (unsigned int i = 0; i < allMatches.size(); i++) {

			Matches matches = allMatches.at(i);
			cv::Mat cAffineMat;
			double cImageScore = 0, cMatrixScore = 0;

			estimateAffineMatrixInLowResImage(kp1, kp2, matches, cAffineMat, roi);

			if (candidates.find(cAffineMat) == candidates.end()) {

				candidates.insert(cAffineMat);

				if (isIdentityMatrix(cAffineMat)) {
					cImageScore = identityImageScore;
					cMatrixScore = identityMatrixScore;
				} else {
					cv::Mat cTransformedImage, cMaskedImage, cRegionMaskedImage;
					transformImage(orgResolution2, cAffineMat , cTransformedImage);
					if(log){
						saveScaledImage("test_orgRes2_"+std::to_string(iteration), cTransformedImage);
					}
					if(lowres_mask2.data) {
						transformMask(orgMask2, cAffineMat, cMaskedImage);
						transformMask(orgRegionMask2, cAffineMat, cRegionMaskedImage);
						if(log){
							saveScaledImage("test_orgRes2_"+std::to_string(iteration)+"m", cMaskedImage);
						}
						cImageScore = getImageDifferenceNormalizedMasked(orgResolution1,
								cTransformedImage, orgMask1, cMaskedImage, orgRegionMask1, cRegionMaskedImage);
					} else {
						//cImageScore = getImageDifference2(image1, cTransformedImage);
						cImageScore = getImageDifferenceNormalized(orgResolution1, cTransformedImage);
					}
					cMatrixScore = getAffineMatrixCost(cAffineMat);
				}

				double cScore = cImageScore + cMatrixScore;

				if (cScore < minScoreSoFar) {
					regInfo.push_back(
							RegistrationInfo(matches, cAffineMat, cImageScore,
									cMatrixScore, minScoreSoFar, true));
					minScoreSoFar = cScore;
					bestMatchIndex = i;
				} else {
					regInfo.push_back(
							RegistrationInfo(matches, cAffineMat, cImageScore,
									cMatrixScore, minScoreSoFar, false));
				}

			} else {
				regInfo.push_back(
						RegistrationInfo(matches, cAffineMat, -1,
								-1, minScoreSoFar, false));
			}

		}

		if (bestMatchIndex >= 0) {
			Matches bestMatches = allMatches.at(bestMatchIndex);
			estimateAffineMatrixInLowResImage(kp1, kp2, bestMatches, affineMat, roi);
			//Mahsa:
			affineMat.at<double>(0,2)= pow(2,iteration) * affineMat.at<double>(0,2);
			affineMat.at<double>(1,2)= pow(2,iteration) * affineMat.at<double>(1,2);
			transformImage(image2, affineMat, image2);
			transformMask(mask2, affineMat, mask2);
			transformMask(region_mask2, affineMat, region_mask2);
		}
	} else { //use all points

		cv::Mat transformedImage, maskedImage, regionMaskedImage;

		estimateAffineMatrixInLowResImage(kp1, kp2, good_matches, affineMat, roi);
		double imageScore;
		transformImage(orgResolution2, affineMat, transformedImage);
		if(lowres_mask2.data) {
			transformMask(orgMask2, affineMat, maskedImage);
			transformMask(orgRegionMask2, affineMat, regionMaskedImage);
			imageScore = getImageDifferenceNormalizedMasked(orgResolution1,
					transformedImage, orgMask1, maskedImage, orgRegionMask1, regionMaskedImage);
		} else {
			//imageScore = getImageDifference2(image1, transformedImage);
			imageScore = getImageDifferenceNormalized(orgResolution1, transformedImage);
		}

		double matrixScore = getAffineMatrixCost(affineMat);
		double cScore = imageScore + matrixScore;

		if (cScore < minScoreSoFar) {
			//Mahsa:
			affineMat.at<double>(0,2)= pow(2, iteration) *  affineMat.at<double>(0,2);
			affineMat.at<double>(1,2)= pow(2, iteration) * affineMat.at<double>(1,2);
			transformImage(image2, affineMat, image2);
			transformMask(mask2, affineMat, mask2);
			transformMask(region_mask2, affineMat, region_mask2);
			regInfo.push_back(
					RegistrationInfo(good_matches, affineMat, imageScore,
							matrixScore, minScoreSoFar, true));
		} else {
			regInfo.push_back(
					RegistrationInfo(good_matches, affineMat, imageScore,
							matrixScore, minScoreSoFar, false));
		}
	}
	if(log){
		logRegisteredImageAndAffineMatrix(affineMat, iteration);
		logTransformation(kp1, kp2, regInfo, iteration);
	}

	//std::cout << affineMat << std::endl;

	return affineMat;
}

void SIFTRegistration::addCoordinatesToDescriptors(KeyPoints kp, cv::Mat &des, int level) {
	cv::Mat xy(des.rows, 2, CV_32F);
	for (unsigned int i = 0; i < kp.size(); i++) {
		xy.at<float>(i, 0) = kp[i].pt.x * pow(2,level);
		xy.at<float>(i, 1) = kp[i].pt.y * pow(2,level); //I think I should not multiply them in pow(...)
	}
	hconcat(des, xy, des);
}



void SIFTRegistration::addScaledCoordinatesToDescriptors(KeyPoints kp, cv::Mat &des,
		cv::Mat scaleMat, int level) {
	cv::Mat xy(des.rows, 2, CV_32F);
	for (unsigned int i = 0; i < kp.size(); i++) {
		xy.at<float>(i, 0) = (kp[i].pt.x * pow(2,level)) * scaleMat.at<double>(0,0)
				+ (kp[i].pt.y * pow(2,level)) * scaleMat.at<double>(1,0) + scaleMat.at<double>(2,0) ;
		xy.at<float>(i, 1) = (kp[i].pt.x * pow(2,level)) * scaleMat.at<double>(0,1)
				+ (kp[i].pt.y * pow(2,level)) * scaleMat.at<double>(1,1) + scaleMat.at<double>(2,1);
	}
	hconcat(des, xy, des);
}



SIFTRegistration::RegistrationInfo::RegistrationInfo(Matches matches,
		cv::Mat affineMatrix, double imageScore, double matrixScore,
		double scoreToBeat, bool better) {
	this->matches = matches;
	this->affineMatrix = affineMatrix;
	this->imageScore = imageScore;
	this->matrixScore = matrixScore;
	this->scoreToBeat = scoreToBeat;
	this->better = better;
}


void SIFTRegistration::initializeParameters() {
	numMatches = 8;
	numSIFTIterations = 4;//8
	SIFTMatchRatio = 0.8f;
}

cv::Mat SIFTRegistration::loadInitialAffineMatrix() {
	return Utils::loadAffineMatrixFromFile(siftInitMatFile);
}

void SIFTRegistration::saveRegisteredImage(const std::string filename) {
	imwrite(filename, image2);
}

void SIFTRegistration::saveScaledImage(const std::string filename, cv::Mat res) {
	std::string name = saveDirectory+"/"+filename+".png";
	imwrite(name , res);
}

void SIFTRegistration::saveSIFTAffineMatrix(const std::string siftMatFile,
		const std::string siftMatLogFile) {

	std::cout << "Saving Mats ... " << std::endl;
    cv::Mat affineMatrix;
	affineMatrix = cv::Mat::eye(3, 3, CV_64F);
	for (int i = siftMats.size() - 1; i >= 0; i--) {
		affineMatrix *= Utils::get3X3MatrixFromAffineMatrix(siftMats.at(i));
	}
	affineMatrix *= Utils::get3X3MatrixFromAffineMatrix(siftInitMat);

	Utils::saveAffineMatrixToFile(affineMatrix, siftMatFile);
	Utils::saveAffineMatrixToFile(siftMats, siftMatLogFile);
}

void SIFTRegistration::saveSIFTAffineMatrixCost(const std::string siftMatCostFile) {

	std::cout << "Saving Affine Mat Cost ... " << std::endl;
    cv::Mat affineMatrix;
	affineMatrix = cv::Mat::eye(3, 3, CV_64F);
	for (int i = siftMats.size() - 1; i >= 0; i--) {
		affineMatrix *= Utils::get3X3MatrixFromAffineMatrix(siftMats.at(i));
	}
	//affineMatrix *= Utils::get3X3MatrixFromAffineMatrix(siftInitMat);
	double cost = getAffineMatrixCost(affineMatrix)/lambda;

	Utils::saveAffineMatrixCostToFile(cost, siftMatCostFile);
}

void SIFTRegistration::logKeyPointsAndDescriptors(const KeyPoints kp1,
		const KeyPoints kp2, const cv::Mat des1, const cv::Mat des2,
		const cv::Mat res1, const cv::Mat res2, const int level) {

	std::ofstream kpfile;
	std::string kpfilename = saveDirectory + "sift_kp_" + std::to_string(level) + ".txt";
	kpfile.open(kpfilename.c_str());
	kpfile << "===================================\n";
	kpfile << "COMPUTER GENERATED DO NOT MODIFY\n";
	kpfile << "===================================\n";
	kpfile << kp1.size() << "\n";
	for (cv::KeyPoint p : kp1) {
		kpfile << p.angle << " " << p.class_id << " " << p.octave << " " << p.pt
				<< " " << p.response << " " << p.size << "\n";
	}
	kpfile << "\n" << kp2.size() << "\n";
	for (cv::KeyPoint p : kp2) {
		kpfile << p.angle << " " << p.class_id << " " << p.octave << " " << p.pt
				<< " " << p.response << " " << p.size << "\n";
	}
	kpfile.close();

	std::ofstream desfile;
	std::string desfilename = saveDirectory + "sift_des_" + std::to_string(level) + ".txt";
	desfile.open(desfilename.c_str());
	desfile << "===================================\n";
	desfile << "COMPUTER GENERATED DO NOT MODIFY\n";
	desfile << "===================================\n";
	desfile << des1 << "\n" << des2 << "\n";
	desfile.close();

	std::string name1 = saveDirectory + "res_" + std::to_string(level) + "_1.tiff";
	cv::imwrite(name1,res1);
	std::string name2 = saveDirectory + "res_" + std::to_string(level) + "_2.tiff";
	cv::imwrite(name2,res2);

	cv::Mat overlay1, overlay2;
	Utils::drawKeyPointsCustom(res1, kp1, overlay1, cv::Scalar::all(0),
			cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS, 2);
	Utils::drawKeyPointsCustom(res2, kp2, overlay2, cv::Scalar::all(0),
			cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS, 2);
	cv::resize(overlay1, overlay1, cv::Size(overlay1.cols/2, overlay1.rows/2));
	cv::resize(overlay2, overlay2, cv::Size(overlay2.cols/2, overlay2.rows/2));
	std::string name3 = saveDirectory + "kp_" + std::to_string(level) + "_1.tiff";
	cv::imwrite(name3,overlay1);
	std::string name4 = saveDirectory + "kp_" + std::to_string(level) + "_2.tiff";
	cv::imwrite(name4,overlay2);
}

void SIFTRegistration::logRegisteredImageAndAffineMatrix(const cv::Mat affineMat, const int level) {

	std::ofstream desfile;
	std::string desfilename = saveDirectory + "affine_matrix_" + std::to_string(level) + ".txt";
	desfile.open(desfilename.c_str());
	desfile << "===================================\n";
	desfile << "COMPUTER GENERATED DO NOT MODIFY\n";
	desfile << "===================================\n";
	desfile << affineMat << "\n";
	desfile.close();

	std::string name1 = saveDirectory + "registered_static_" + std::to_string(level) + ".tiff";
	cv::imwrite(name1,image1);
	std::string name2 = saveDirectory + "registered_moving_" + std::to_string(level) + ".tiff";
	cv::imwrite(name2,image2);
}

void SIFTRegistration::logCropping(const cv::Mat image1, const cv::Mat  image2,
		const cv::Mat mask1, const cv::Mat  mask2, cv::Rect roi, int level){
	cv::Scalar color = cv::Scalar(0, 25, 0);
	cv::Mat copyImg1, copyImg2, copyM1, copyM2;
	copyImg1 = image1.clone();
	copyImg2 = image2.clone();
	copyM1 = mask1.clone();
	copyM2 = mask2.clone();
	cv::rectangle(copyImg1, roi, color,3, 8, 0);
	cv::rectangle(copyImg2, roi, color,3, 8, 0);
	cv::rectangle(copyM1, roi, color,3, 8, 0);
	cv::rectangle(copyM2, roi, color,3, 8, 0);
	std::string name1 = saveDirectory + "roi_static_image_" + std::to_string(level) + ".tiff";
	cv::imwrite(name1,copyImg1);
	std::string name2 = saveDirectory + "roi_moving_image_" + std::to_string(level) + ".tiff";
	cv::imwrite(name2,copyImg2);
	std::string name3 = saveDirectory + "roi_static_mask_" + std::to_string(level) + ".tiff";
	cv::imwrite(name3,copyM1);
	std::string name4 = saveDirectory + "roi_moving_mask_" + std::to_string(level) + ".tiff";
	cv::imwrite(name4,copyM2);
}

void SIFTRegistration::logMatches(const Matches good_matches,
		const Matches matches_outliers_removed, const Matches selected_matches,
		const KeyPoints kp1, const KeyPoints kp2, const int level) {
	std::ofstream mfile;
	std::string mfilename = saveDirectory + "sift_matches_" + std::to_string(level) + ".txt";
	mfile.open(mfilename.c_str());
	mfile << "===================================\n";
	mfile << "COMPUTER GENERATED DO NOT MODIFY\n";
	mfile << "===================================\n";
	mfile << good_matches.size() << "\n";

	for (cv::DMatch m : good_matches) {
		mfile << m.imgIdx << " " << kp1[m.queryIdx].pt.x << " "
				<< kp1[m.queryIdx].pt.y << " " << kp2[m.trainIdx].pt.x << " "
				<< kp2[m.trainIdx].pt.y << " " << m.distance << "\n";
	}
	mfile << "\n" << matches_outliers_removed.size() << "\n";
	for (cv::DMatch m : matches_outliers_removed) {
		mfile << m.imgIdx << " " << kp1[m.queryIdx].pt.x << " "
				<< kp1[m.queryIdx].pt.y << " " << kp2[m.trainIdx].pt.x << " "
				<< kp2[m.trainIdx].pt.y << " " << m.distance << "\n";
	}
	mfile << "\n" << selected_matches.size() << "\n";
	for (cv::DMatch m : selected_matches) {
		mfile << m.imgIdx << " " << kp1[m.queryIdx].pt.x << " "
				<< kp1[m.queryIdx].pt.y << " " << kp2[m.trainIdx].pt.x << " "
				<< kp2[m.trainIdx].pt.y << " " << m.distance << "\n";
	}
	mfile.close();

	cv::Mat overlay1, overlay2, overlay3;
	Utils::drawMatchesCustom(resolution1, kp1, resolution2, kp2, good_matches, overlay1,
			cv::Scalar::all(0), cv::Scalar::all(0), std::vector<char>(),
			cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS, 2);
	Utils::drawMatchesCustom(resolution1, kp1, resolution2, kp2, matches_outliers_removed,
			overlay2, cv::Scalar::all(0), cv::Scalar::all(0),
			std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,
			2);
	Utils::drawMatchesCustom(resolution1, kp1, resolution2, kp2, selected_matches,
			overlay3, cv::Scalar::all(0), cv::Scalar::all(0),
			std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,
			2);

	cv::resize(overlay1, overlay1, cv::Size(overlay1.cols/2, overlay1.rows/2));
	cv::resize(overlay2, overlay2, cv::Size(overlay2.cols/2, overlay2.rows/2));
	cv::resize(overlay3, overlay3, cv::Size(overlay3.cols/2, overlay3.rows/2));

	std::string name1 = saveDirectory + "matches_good_" + std::to_string(level) + ".tiff";
	cv::imwrite(name1, overlay1);
	std::string name2 = saveDirectory + "matches_no_outliers_" + std::to_string(level)+ ".tiff";
	cv::imwrite(name2, overlay2);
	std::string name3 = saveDirectory + "matches_selected_" + std::to_string(level)	+ ".tiff";
	cv::imwrite(name3, overlay3);
}

void SIFTRegistration::logChosenMatches(const Matches selected_matches,
		const KeyPoints kp1, const KeyPoints kp2, const int level) {

	cv::Mat overlay3;
	Utils::drawMatchesCustom(resolution1, kp1, resolution2, kp2, selected_matches,
			overlay3, cv::Scalar::all(0), cv::Scalar::all(0),
			std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,
			2);

	std::string name3 = saveDirectory + "matches_chosen_" + std::to_string(level)	+ ".tiff";
	cv::imwrite(name3, overlay3);
}

void SIFTRegistration::logTransformation(const KeyPoints kp1,
		const KeyPoints kp2, const std::vector<RegistrationInfo> regInfo,
		const int blur) {
	std::ofstream mfile;
	std::string mfilename = saveDirectory + "sift_transformation_" + std::to_string(blur)+ ".txt";
	mfile.open(mfilename.c_str());
	mfile << "===================================\n";
	mfile << "COMPUTER GENERATED DO NOT MODIFY\n";
	mfile << "===================================\n";

	for (RegistrationInfo r : regInfo) {
		for (cv::DMatch m : r.matches) {
			mfile << m.imgIdx << " " << kp1[m.queryIdx].pt.x << " "
					<< kp1[m.queryIdx].pt.y << " " << kp2[m.trainIdx].pt.x
					<< " " << kp2[m.trainIdx].pt.y << " " << m.distance << "\n";
		}
		for (int i = 0; i < r.affineMatrix.rows; i++) {
			for (int j = 0; j < r.affineMatrix.cols; j++)
				mfile << r.affineMatrix.at<double>(i, j) << " ";
		}
		mfile << "\n" << r.imageScore << " " << r.matrixScore << " "
				<< r.imageScore + r.matrixScore << " " << r.scoreToBeat << " "
				<< std::boolalpha << r.better << "\n\n";
	}

	mfile.close();
}


void SIFTRegistration::transformMask(const cv::Mat inputImage,
		const cv::Mat affineMat, cv::Mat &outputImage) {

	warpAffine(inputImage, outputImage, affineMat, outputImage.size());
	Utils::floodCorners(outputImage, cv::Scalar(0.0, 0.0, 0.0));
}

void SIFTRegistration::getGoodMatches(Matches &good_matches, cv::Mat des1,
		cv::Mat des2) {
	cv::FlannBasedMatcher matcher;
	std::vector<Matches> knnMatches;
	matcher.knnMatch(des1, des2, knnMatches, 2);
	good_matches.reserve(knnMatches.size());
	for (size_t i = 0; i < knnMatches.size(); ++i) {
		if (knnMatches[i].size() < 2)
			continue;
		const cv::DMatch &m1 = knnMatches[i][0];
		const cv::DMatch &m2 = knnMatches[i][1];

		if (m1.distance < SIFTMatchRatio * m2.distance) //0.8 as per Lowe's paper
			good_matches.push_back(m1);
	}
}

void SIFTRegistration::removeUniqueMatches(KeyPoints kp1, KeyPoints kp2,
		Matches &good_matches) {

	std::sort(good_matches.begin(), good_matches.end(),
			[](cv::DMatch a, cv::DMatch b) {
				return a.distance < b.distance;
			});

	std::set<int> toRemove;
	for (unsigned int i = 0; i < good_matches.size(); i++) {
		for (unsigned int j = i + 1; j < good_matches.size(); j++) {
			if ((kp1[good_matches[i].queryIdx].pt.x
					== kp1[good_matches[j].queryIdx].pt.x
					&& kp1[good_matches[i].queryIdx].pt.y
							== kp1[good_matches[j].queryIdx].pt.y)
					|| (kp2[good_matches[i].trainIdx].pt.x
							== kp2[good_matches[j].trainIdx].pt.x
							&& kp2[good_matches[i].trainIdx].pt.y
									== kp2[good_matches[j].trainIdx].pt.y)) {
				toRemove.insert(j);
			}
		}
	}

	for (std::set<int>::reverse_iterator it = toRemove.rbegin();
			it != toRemove.rend(); it++)
		good_matches.erase(good_matches.begin() + *it);
}

void SIFTRegistration::removeBadFeatureDistance(KeyPoints kp1, KeyPoints kp2,
		Matches &good_matches, int distance) {

	sort(good_matches.begin(), good_matches.end(),
			[](cv::DMatch a, cv::DMatch b) {
				return a.distance < b.distance;
			});
	Matches::iterator it = good_matches.end();
	for (Matches::iterator i = good_matches.begin(); i != good_matches.end();
			i++) {
		if (i->distance > distance) {
			it = i;
			break;
		}
	}

	if (it != good_matches.end())
		good_matches.erase(it, good_matches.end());
}

void SIFTRegistration::removeBadEuclideanDistance(KeyPoints kp1, KeyPoints kp2,
		Matches &good_matches, int distance) {

	sort(good_matches.begin(), good_matches.end(),
			[](cv::DMatch a, cv::DMatch b) {
				return a.distance < b.distance;
			});
	std::set<int> toRemove;
	for (unsigned int i = 0; i < good_matches.size(); i++) {
		cv::Point s = kp1[good_matches[i].queryIdx].pt;
		cv::Point o = kp2[good_matches[i].trainIdx].pt;
		double dx = sqrt((s.x - o.x) * (s.x - o.x) + (s.y - o.y) * (s.y - o.y));
		if (dx > distance) {
			toRemove.insert(i);
		}
	}
	for (std::set<int>::reverse_iterator it = toRemove.rbegin();
			it != toRemove.rend(); it++) {
		good_matches.erase(good_matches.begin() + *it);
	}
}

void SIFTRegistration::rankAndRemoveFeature(KeyPoints kp1, KeyPoints kp2,
		Matches &good_matches, int numPt) {

	sort(good_matches.begin(), good_matches.end(),
			[](cv::DMatch a, cv::DMatch b) {
				return a.distance < b.distance;
			});

	if (good_matches.size() > numPt) {
		good_matches.erase(good_matches.begin() + numPt, good_matches.end());
	}
}

void SIFTRegistration::rankAndRemoveEuclidean(KeyPoints kp1, KeyPoints kp2,
		Matches &good_matches, int numPt) {

	sort(good_matches.begin(), good_matches.end(),
			[kp1,kp2](cv::DMatch a, cv::DMatch b) {
				cv::Point as = kp1[a.queryIdx].pt;
				cv::Point ao = kp2[a.trainIdx].pt;
				double adx = sqrt((as.x-ao.x) * (as.x-ao.x)
						+ (as.y-ao.y) * (as.y-ao.y));
				cv::Point bs = kp1[b.queryIdx].pt;
				cv::Point bo = kp2[b.trainIdx].pt;
				double bdx = sqrt((bs.x-bo.x) * (bs.x-bo.x)
						+ (bs.y-bo.y) * (bs.y-bo.y));
				return adx < bdx;

			});

	if (good_matches.size() > numPt){
		good_matches.erase(good_matches.begin() + numPt, good_matches.end());
	}
}

//Sum of Pixel difference squared
double SIFTRegistration::getImageDifference(const cv::Mat im1,
		const cv::Mat im2) {


	cv::Mat int1, int2;
	im1.convertTo(int1, CV_64FC3);
	im2.convertTo(int2, CV_64FC3);
    cv::Mat diff2, sqr;
	diff2 = int1 - int2;
	sqr = diff2.mul(diff2);

	double difference = sum(sum(sqr))[0];

	return sqrt(difference);

}

double SIFTRegistration::getImageDifferenceNormalized(const cv::Mat im1,
		const cv::Mat im2) {

	int counter = 0;
	double sum = 0;

	for (int i = 0; i < im1.rows; i++) {
		for (int j = 0; j < im1.cols; j++) {
			cv::Vec3b v1 = im1.at<cv::Vec3b>(i, j);
			cv::Vec3b v2 = im2.at<cv::Vec3b>(i, j);


			if (v1[0] == 255
				&& v2[0] == 255){
				continue;
			} else {
				//get pixel squared difference and add to counter
				double temp1 = (v1[0] - v2[0]) * (v1[0] - v2[0]);
				sum += sqrt(temp1);
				counter++;
			}
		}
	}


	double SSD = 0;
	if (counter!=0)
		SSD =  sum/(double)counter;
	else
		SSD =  std::numeric_limits<double>::infinity();

	//std::cout << "SSD: "  <<  SSD << " count:" << counter <<"\n";
	return SSD;
}

//Sum of Pixel difference squared
double SIFTRegistration::getImageDifference(const cv::Mat im1,
		const cv::Mat im2, const cv::Mat affineMatrix) {

	double cost1 = getImageDifference(im1, im2);
	cv::Mat affine = Utils::get3X3MatrixFromAffineMatrix(affineMatrix);
	cv::Mat diff;
	diff = affine - cv::Mat::eye(3, 3, CV_64FC1);
	double cost2 = getFrobeniusNorm(diff);

	double totalCost = cost1 + lambda * cost2;

	//std::cout<<cost1<<"\t"<<cost2<<"\t"<<totalCost<<std::endl;

	return totalCost;
}

//Sum of Pixel difference squared
double SIFTRegistration::getImageDifference(const cv::Mat im1,
		const cv::Mat im2, const cv::Mat affineMatrix,
		const double identityScore) {

	double cost1 = 0, cost2 = 0;
	if (sum(sum(affineMatrix - cv::Mat::eye(2, 3, CV_64FC1)))[0] == 0) {
		cost1 = identityScore;
		cost2 = 0;
	} else {
		cost1 = getImageDifference(im1, im2);
		cv::Mat affine = Utils::get3X3MatrixFromAffineMatrix(affineMatrix);
		cv::Mat diff;
		diff = affine - cv::Mat::eye(3, 3, CV_64FC1);
		cost2 = getFrobeniusNorm(diff);
	}

	double totalCost = cost1 + lambda * cost2;

	//std::cout<<cost1<<"\t"<<cost2<<"\t"<<totalCost<<std::endl;

	return totalCost;
}

double SIFTRegistration::getFrobeniusNorm(cv::Mat mat) {
	cv::Mat matT;
	transpose(mat, matT);
	double norm = sqrt(trace(mat * matT)[0]);

	return norm;
}

void SIFTRegistration::combination(int offset, int K, Matches &good_matches,
		std::vector<Matches> &allmatches, Matches &currentMatch) {

	if (K == 0) {
		allmatches.push_back(Matches(currentMatch));
		return;
	}

	for (unsigned int i = offset; i <= good_matches.size() - K; ++i) {
		currentMatch.push_back(good_matches.at(i));
		combination(i + 1, K - 1, good_matches, allmatches, currentMatch);
		currentMatch.pop_back();
	}

}

cv::Mat SIFTRegistration::performRigidTransform(KeyPoints kp1, KeyPoints kp2,
		Matches good_matches, cv::Mat inputImage, cv::Mat &outputImage) {

	std::vector<cv::Point2f> pt1, pt2;

	for (unsigned int i = 0; i < good_matches.size(); i++) {
		pt1.push_back(kp1[good_matches[i].queryIdx].pt);
		pt2.push_back(kp2[good_matches[i].trainIdx].pt);
	}

	cv::Mat H = cv::estimateRigidTransform(pt2, pt1, false);

	if (H.rows == 0) {
		inputImage.copyTo(outputImage);
		H = cv::Mat::zeros(cv::Size(3, 2), CV_64FC1);
		H.at<double>(0, 0) = 1.0;
		H.at<double>(1, 1) = 1.0;
		return H;
	}

	warpAffine(inputImage, outputImage, H, outputImage.size(), cv::INTER_LANCZOS4+CV_WARP_FILL_OUTLIERS);

	floodFill(outputImage, cv::Point(0, 0), cv::Scalar(255.0, 255.0, 255.0), 0,
			cv::Scalar(0.0, 0.0, 0.0), cv::Scalar(254.0, 254.0, 254.0));
	floodFill(outputImage, cv::Point(0, outputImage.rows - 1),
			cv::Scalar(255.0, 255.0, 255.0), 0, cv::Scalar(0.0, 0.0, 0.0),
			cv::Scalar(254.0, 254.0, 254.0));
	floodFill(outputImage, cv::Point(outputImage.cols - 1, 0),
			cv::Scalar(255.0, 255.0, 255.0), 0, cv::Scalar(0.0, 0.0, 0.0),
			cv::Scalar(254.0, 254.0, 254.0));
	floodFill(outputImage,
			cv::Point(outputImage.cols - 1, outputImage.rows - 1),
			cv::Scalar(255.0, 255.0, 255.0), 0, cv::Scalar(0.0, 0.0, 0.0),
			cv::Scalar(254.0, 254.0, 254.0));

	return H;
}

void SIFTRegistration::estimateAffineMatrix(const KeyPoints kp1,
		const KeyPoints kp2, const Matches good_matches, cv::Mat &affineMat) {

	std::vector<cv::Point2f> pt1, pt2;

	for (unsigned int i = 0; i < good_matches.size(); i++) {
		pt1.push_back(kp1[good_matches[i].queryIdx].pt);
		pt2.push_back(kp2[good_matches[i].trainIdx].pt);
	}

	affineMat = cv::estimateRigidTransform(pt2, pt1, false);

	if (affineMat.rows == 0) {
		affineMat = cv::Mat::zeros(cv::Size(3, 2), CV_64FC1);
		affineMat.at<double>(0, 0) = 1.0;
		affineMat.at<double>(1, 1) = 1.0;
	}
}

void SIFTRegistration::estimateAffineMatrixInLowResImage(const KeyPoints kp1,
		const KeyPoints kp2, const Matches good_matches, cv::Mat &affineMat, cv::Rect roi) {

	std::vector<cv::Point2f> pt1, pt2;

	for (unsigned int i = 0; i < good_matches.size(); i++) {
		cv::Point2d point1 = kp1[good_matches[i].queryIdx].pt;
		cv::Point2d point2 = kp2[good_matches[i].trainIdx].pt;
		point1.x = point1.x + roi.x;
		point1.y = point1.y + roi.y;
		point2.x = point2.x + roi.x;
		point2.y = point2.y + roi.y;
		pt1.push_back(point1);
		pt2.push_back(point2);
	}

	affineMat = cv::estimateRigidTransform(pt2, pt1,false); //getAffineTrasform,estimateAffinePartial2D(pt2, pt1);

	if (affineMat.rows == 0) {
		affineMat = cv::Mat::zeros(cv::Size(3, 2), CV_64FC1);
		affineMat.at<double>(0, 0) = 1.0;
		affineMat.at<double>(1, 1) = 1.0;
	}
}

void SIFTRegistration::estimateAffineMatrixInLowResImageScaled(const KeyPoints kp1,
		const KeyPoints kp2, const Matches good_matches, cv::Mat &affineMat, cv::Rect roi, cv::Mat scale) {

	std::vector<cv::Point2f> pt1, pt2;

	for (unsigned int i = 0; i < good_matches.size(); i++) {
		cv::Point2d point1 = kp1[good_matches[i].queryIdx].pt;
		cv::Point2d point2 = kp2[good_matches[i].trainIdx].pt;
		point1.x = (point1.x + roi.x);
		point1.y = (point1.y + roi.y);
		point2.x = scale.at<double>(0,0)*(point2.x + roi.x);
		point2.y = scale.at<double>(1,1)*(point2.y + roi.y);
		pt1.push_back(point1);
		pt2.push_back(point2);
	}

	affineMat = cv::estimateRigidTransform(pt2, pt1,false); //getAffineTrasform,estimateAffinePartial2D(pt2, pt1);

	if (affineMat.rows == 0) {
		affineMat = cv::Mat::zeros(cv::Size(3, 2), CV_64FC1);
		affineMat.at<double>(0, 0) = 1.0;
		affineMat.at<double>(1, 1) = 1.0;
	}
}


void SIFTRegistration::transformImage(const cv::Mat inputImage,
		const cv::Mat affineMat, cv::Mat &outputImage) {

	warpAffine(inputImage, outputImage, affineMat, outputImage.size(), cv::INTER_LANCZOS4+CV_WARP_FILL_OUTLIERS);
	Utils::floodCorners(outputImage, cv::Scalar(255.0, 255.0, 255.0),
			cv::Scalar(0.0, 0.0, 0.0), cv::Scalar(254.0, 254.0, 254.0));
}

double SIFTRegistration::getImageDifference2(const cv::Mat image1,
		const cv::Mat image2) {

	cv::Mat int1, int2;
	image1.convertTo(int1, CV_64FC3);
	image2.convertTo(int2, CV_64FC3);

	cv::Mat diff2, sqr;
	diff2 = int1 - int2;
	sqr = diff2.mul(diff2);

	double difference = sqrt(sum(sum(sqr))[0]);

	return difference;
}

double SIFTRegistration::getAffineMatrixCost(const cv::Mat affineMatrix) {
	cv::Mat affineMatrix3X3 = Utils::get3X3MatrixFromAffineMatrix(affineMatrix);
	cv::Mat diff;
	diff = affineMatrix3X3 - cv::Mat::eye(3, 3, CV_64FC1);
	double affineMatrixCost = getFrobeniusNorm(diff);
	return lambda * affineMatrixCost;
}


bool SIFTRegistration::isIdentityMatrix(const cv::Mat affineMatrix) {
	return sum(sum(affineMatrix - cv::Mat::eye(2, 3, CV_64FC1)))[0] == 0;
}

#endif /* SIFT_HPP_ */
