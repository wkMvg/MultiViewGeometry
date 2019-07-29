#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

template<typename t>
t d2r(t d)
{
	return d * 3.1415 / 180;
}

/*北东地--前右下*/
Matrix<double, 3, 3> eula2roation(const Matrix<double, 3, 1> eula)
{
	Matrix<double, 3, 3> rx = Matrix<double, 3, 3>::Identity();
	Matrix<double, 3, 3> ry = Matrix<double, 3, 3>::Identity();
	Matrix<double, 3, 3> rz = Matrix<double, 3, 3>::Identity();

	double x = eula(0, 0);
	double y = eula(1, 0);
	double z = eula(2, 0);

	rx(1, 1) = cos(d2r<double>(x)); rx(1, 2) = sin(d2r<double>(x));
	rx(2, 1) = -sin(d2r<double>(x)); rx(2, 2) = cos(d2r<double>(x));

	ry(0, 0) = cos(d2r<double>(y)); ry(0, 2) = -sin(d2r<double>(y));
	ry(2, 0) = sin(d2r<double>(y)); ry(2, 2) = cos(d2r<double>(y));

	rz(0, 0) = cos(d2r<double>(z)); rz(0, 1) = sin(d2r<double>(z));
	rz(1, 0) = -sin(d2r<double>(z)); rz(1, 1) = cos(d2r<double>(z));

	return rx * ry * rz;
}

Eigen::Matrix<double, 2, 1> reprojectToImg(const Matrix<double, 3, 1> _3d_point,
	const Matrix<double, 3, 3> k,
	const Matrix<double, 3, 1> eula,
	const Matrix<double,3,1> camPos,
	const vector<double>& distortionCoeff
)
{
	const double k1 = distortionCoeff[0];
	const double k2 = distortionCoeff[1];
	const double k3 = distortionCoeff[2];
	const double p1 = distortionCoeff[3];
	const double p2 = distortionCoeff[4];

	Matrix<double, 3, 3> rotation = eula2roation(eula);

	// 从cgcs2000坐标系转换到机体坐标系
	Matrix<double, 3, 1> _3d_point_body = rotation * (_3d_point - camPos);

	// 飞机的机体坐标系与相机坐标系的x和y轴相反,再从机体坐标系转换到相机坐标系
	Matrix<double, 3, 3> body2cam = Matrix<double, 3, 3>::Zero();
	body2cam(0, 1) = 1;
	body2cam(1, 0) = -1;
	body2cam(2, 2) = 1;

	Matrix<double, 3, 1> _3d_point_cam = body2cam * _3d_point_body;

	double x1 = _3d_point_cam(0, 0) / _3d_point_cam(2, 0);
	double y1 = _3d_point_cam(1, 0) / _3d_point_cam(2, 0);
	// 利用畸变因子，计算畸变后的位置
	double r2 = x1 * x1 + y1 * y1;
	double x_distort = x1 * (1 + k1 * r2 + k2 * r2 * r2) + 2 * p1 * x1 * y1 + p2 * (r2 + 2 * x1 * x1);
	double y_distort = y1 * (1 + k1 * r2 + k2 * r2 * r2) + p1 * (r2 + 2 * y1 * y1) + 2 * p2 * x1 * y1;

	double u_distort = k(0, 0) * x_distort + k(0, 2);
	double v_distort = k(0, 0) * y_distort + k(1, 2);

	double u_undistort = k(0, 0) * x1 + k(0, 2);
	double v_undistort = k(0, 0) * y1 + k(1, 2);

	return Matrix<double, 2, 1>(u_distort, v_distort);
}

int main()
{
	//三维点坐标
	Matrix<double, 3, 1> _3d_point;
	_3d_point << 3112837.371540, 386094.250464, -928.534871;

	//相机内参
	Matrix<double, 3, 3> k;   
	k << 8012.5247, 0, 3987.1933,
		0, 8012.5247, 2642.38017,
		0, 0, 1;

	//畸变因子
	const vector<double> distortCoeff{ 0.0550482 ,-0.225193 ,0.0018439 ,0.000593564 ,0.000115175 };

	//相机欧拉角
	Matrix<double, 3, 1> eula_327;
	eula_327 << 1.74, 7.195, 51.85;

	//相机位置
	Matrix<double, 3, 1> position_327;
	position_327 << 3112800.233901, 386155.629066, -1611.633999;
	
	Matrix<double, 2, 1> pix_327 = reprojectToImg(_3d_point, k, eula_327, position_327, distortCoeff);
}
