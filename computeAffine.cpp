#include "mvg.hpp"

int main()
{
    string dir1 = "F:\\data\\300\\image\\DSC00003.JPG";
    string dir2 = "F:\\data\\300\\image\\DSC00005.JPG";
    int nfeatures = 2000;
	float matchRatio = 0.6;
	int ransacIterations = 200;
    
	mvg* doubleView = new mvg(dir1, dir2, nfeatures, matchRatio, ransacIterations);
}
