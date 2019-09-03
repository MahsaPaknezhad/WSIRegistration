
#include <cv.h>
#include <highgui.h>
#include "TranslateScale.hpp"
#include "Rotate.hpp"
#include "SIFTRegistration.hpp"
#include "Register.hpp"
#include "RoiInfo.hpp"
#include "Utils.hpp"
#include "chrono"

using namespace cv;


int main (int argc, char** argv) {

    bool log = false;
	std::string cases [] = {"Patient1"};
	std::string patients [] = {"Case1"};
	int numImages [] = {100};
	int firstSlice [] = {1};
	std::string IDS [] = {"101"}; // does not matter just an indicator
	double lambdas [] = {0.02}; // the weight for the affine matrix cost function
	auto start = std::chrono::steady_clock::now();
	for(int i=0; i< 1; i++) {
		std::string id = IDS[0];
		double lambda = lambdas[0];
		std::string mumfordAppLocation = "/home/images/"+cases[i]+"/gray_levels.sh";
		std::string roiSelectionApp = "/home/images/"+cases[i]+"/setRegionOfInterest.py";
		Register r("/home/images/"+cases[i]+"/"+patients[i], patients[i], id, lambda, firstSlice[i], numImages[i],
				mumfordAppLocation, roiSelectionApp, log);
	}
	auto end = std::chrono::steady_clock::now();
	std::cout << "Duration in seconds: " << std::chrono::duration_cast<std::chrono::seconds>(end-start).count();





}


