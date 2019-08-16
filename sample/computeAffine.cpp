#include "MVG.hpp"

int main()
{
	/*string dir1 = "F:\\data\\5.11\\image\\DSC00483.JPG";
    string dir2 = "F:\\data\\5.11\\image\\DSC00487.JPG";*/
	string dir1 = "000000l.png";
	string dir2 = "000000r.png";
	/*string dir1 = "1.jpg";
	string dir2 = "2.jpg";*/

    int nfeatures = 100000;
	float matchRatio = 0.6;
	int ransacIterations = 500;

    Eigen::Matrix<double,3,3> intrinsic = Eigen::Matrix<double,3,3>::Identity();
	/*intrinsic(0, 0) = 7158.9282;	intrinsic(1, 1) = 7158.9282;
	intrinsic(0, 2) = 3680;     intrinsic(1, 2) = 2456;*/
	intrinsic(0, 0) = 718.856;	intrinsic(1, 1) = 718.856;
	intrinsic(0, 2) = 607.1928;	intrinsic(1, 2) = 185.2157;

	//wkMvg::mvg* doubleView = new wkMvg::mvg(dir1, dir2, nfeatures, matchRatio, ransacIterations, intrinsic, wkMvg::pattern::homography);
	wkMvg::mvg* doubleViewFunda = new wkMvg::mvg(dir1, dir2, nfeatures, matchRatio, ransacIterations, intrinsic, wkMvg::pattern::fundamental);
	doubleViewFunda->savePly("vertex.ply");
}
