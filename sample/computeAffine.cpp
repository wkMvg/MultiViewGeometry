#include "MVG.hpp"

int main()
{
    string dir1 = "DSC00003.JPG";
    string dir2 = "DSC00005.JPG";
    int nfeatures = 40000;
	float matchRatio = 0.6;
	int ransacIterations = 500;

    Eigen::Matrix<double,3,3> intrinsic = Eigen::Matrix<double,3,3>::Identity();
	wkMvg::mvg* doubleView = new wkMvg::mvg(dir1, dir2, nfeatures, matchRatio, ransacIterations, intrinsic, wkMvg::pattern::homography);
}
