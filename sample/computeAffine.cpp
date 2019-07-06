#include "MVG.hpp"

int main()
{
    string dir1 = "DSC00003.JPG";
    string dir2 = "DSC00005.JPG";
    int nfeatures = 40000;
	float matchRatio = 0.6;
	int ransacIterations = 500;
    
	mvg* doubleView = new mvg(dir1, dir2, nfeatures, matchRatio, ransacIterations);
}
