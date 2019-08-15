#include "MVG.hpp"

using namespace wkMvg;

template<typename t>
inline t max(const vector<t> data)
{
    t maxNum = 0;
    
    for(int i = 0; i<data.size(); i++)
    {
        if(maxNum < data[i])
            maxNum = data[i];
        else
            continue;
    }
    return maxNum;
}

//@param dir1,dir2 输入图像的地址
//@param nfeatures 检测到图像特征点的数量
//该构造函数用来计算仿射变换
mvg::mvg(string dir1, string dir2, const int nfeatures, const float ratio,
        const int maxIterations, Eigen::Matrix<double,3,3> intrinsicK, pattern flag)
{
    mImgl = imread(dir1);
    mImgr = imread(dir2);

    mIntrinsicK = intrinsicK;

    computeFeat(nfeatures);
#ifdef _DEBUG
    showKeyPoint();
#endif
    computeMatches(ratio);
#ifdef  _DEBUG
	showMatch();
#endif //  _DEBUG

	if (mMatches.size() < 100)
		cerr << "Attention!!! There is no enough matches to compute affine matrix!!!\n";

    if(flag == homography)
    {
        normalize();
        ransacAffine(maxIterations);
        computeHomo_leastSquare();
    }
    else if(flag == fundamental)
    {
		normalize();
        ransacFundamental(maxIterations);
        reconstructF();
    }
}

mvg::~mvg()
{
    mKeypointl.clear();
    mKeypointr.clear();
    mMatches.clear();
    mKeypointl_r.clear();
    mKeypointl_r_norm.clear();
    mP3d.clear();
    mIsInliers_Homo.clear();
    mIsInliers_Funda.clear();
    mIsTriangulate.clear();
}

void mvg::computeFeat(const int nfeatures)
{
    Ptr<ORB> orbFeat = ORB::create(nfeatures);
    orbFeat->detectAndCompute(mImgl, Mat(), mKeypointl, mDescl);
    orbFeat->detectAndCompute(mImgr, Mat(), mKeypointr, mDescr);
}

//@param ratio 判断最优匹配与次优匹配距离的比例，比例越小说明两个匹配对差别越大，该匹配越可信
void mvg::computeMatches(const float ratio)
{
    Ptr<FlannBasedMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12,20,1));
    vector<vector<DMatch>> matchesNN;
    matcher->knnMatch(mDescl, mDescr, matchesNN, 2);
#pragma omp parallel for
    for(int i = 0; i < matchesNN.size(); i++)
    {
		if (matchesNN[i].size() == 2)
		{
			DMatch& bestMatch = matchesNN[i][0];
			DMatch& betterMatch = matchesNN[i][1];

			if (bestMatch.distance < ratio * betterMatch.distance)
			{
				mMatches.push_back(bestMatch);
			}
		}
    }

    for(int i = 0; i < mMatches.size(); i++)
    {
        KeyPoint keypoint1 = mKeypointl[mMatches[i].queryIdx];
        KeyPoint keypoint2 = mKeypointr[mMatches[i].trainIdx];
        mKeypointl_r.push_back(make_pair(keypoint1, keypoint2));
    }
}

/*显示检测到的关键点在图像上*/
void mvg::showKeyPoint()
{
    if(mKeypointl.size() == 0 || mKeypointr.size() == 0)
    {
        cerr<<"maybe there is no enough keypoints\n";
    }

    Mat keypointImgl,keypointImgr;

	Scalar color(255.*rand() / RAND_MAX, 255.*rand() / RAND_MAX, 255.*rand() / RAND_MAX);

    drawKeypoints(mImgl,mKeypointl,keypointImgl,Scalar::all(-1));
    drawKeypoints(mImgr,mKeypointr,keypointImgr,Scalar::all(-1));

    imshow("keypointsL",keypointImgl);
    imshow("keypointsR",keypointImgr);

    imwrite("keypointsL.tif",keypointImgl);
    imwrite("keypointsR.tif",keypointImgr);
    waitKey();
}

/*显示图像间的匹配 */
void mvg::showMatch()
{
    if(mMatches.size() == 0)
    {
        cerr << "Maybe there is no enough match\n";
    }
    Mat imgMatch;
	Scalar color(255.*rand() / RAND_MAX, 255.*rand() / RAND_MAX, 255.*rand() / RAND_MAX);
	drawMatches(mImgl, mKeypointl, mImgr, mKeypointr,
		mMatches, imgMatch, Scalar::all(-1));
    imshow("matches",imgMatch);
    imwrite("matches.tif",imgMatch);
    waitKey();
}

/*归一化 x = (x - ex)/dev 
  均值 ex = total / N
  方差 sitax = |x-ex| / N */
void mvg::normalize()
{
    float meanx1 = 0, meany1 = 0, meanx2 = 0, meany2 = 0;
    float meanDevx1 = 0, meanDevy1 = 0, meanDevx2 = 0, meanDevy2 = 0;
    int N = mKeypointl_r.size();

    for(int i = 0; i < N; i++)
    {
        meanx1 += mKeypointl_r[i].first.pt.x;
        meany1 += mKeypointl_r[i].first.pt.y;

        meanx2 += mKeypointl_r[i].second.pt.x;
        meany2 += mKeypointl_r[i].second.pt.y;
    }
    meanx1 /= N;
    meany1 /= N;
    meanx2 /= N;
    meany2 /= N;        

    for(int i = 0; i < N; i++)
    {
        KeyPoint keypoint1 = mKeypointl_r[i].first;
        KeyPoint keypoint2 = mKeypointl_r[i].second;

        keypoint1.pt.x -= meanx1;
        keypoint1.pt.y -= meany1;
        keypoint2.pt.x -= meanx2;
        keypoint2.pt.y -= meany2;

        meanDevx1 += fabs(keypoint1.pt.x);
        meanDevy1 += fabs(keypoint1.pt.y);
        meanDevx2 += fabs(keypoint2.pt.x);
        meanDevy2 += fabs(keypoint2.pt.y);

        mKeypointl_r_norm.push_back(make_pair(keypoint1, keypoint2));
    }

    meanDevx1 /= N;
    meanDevy1 /= N;
    meanDevx2 /= N;
    meanDevy2 /= N;

    for(int i = 0; i < N; i++)
    {
        KeyPoint& keypoint1 = mKeypointl_r_norm[i].first;
        KeyPoint& keypoint2 = mKeypointl_r_norm[i].second;

        keypoint1.pt.x /= meanDevx1;
        keypoint1.pt.y /= meanDevy1;
        keypoint2.pt.x /= meanDevx2;
        keypoint2.pt.y /= meanDevy2;
    }
	mSim3Nl = Eigen::Matrix<double, 3, 3>::Identity();
	mSim3Nr = Eigen::Matrix<double, 3, 3>::Identity();
    mSim3Nl(0,0) = 1 / meanDevx1;    mSim3Nl(0,2) = -meanx1 / meanDevx1;
    mSim3Nl(1,1) = 1 / meanDevy1;    mSim3Nl(1,2) = -meany1 / meanDevy1;
    
    mSim3Nr(0,0) = 1 / meanDevx2;    mSim3Nr(0,2) = -meanx2 / meanDevx2;
    mSim3Nr(1,1) = 1 / meanDevy2;    mSim3Nr(1,2) = -meany2 / meanDevy2;
}

//@param pairs 用来计算单应矩阵的特征点匹配对
//@param currHomo 计算得到的单应矩阵
void mvg::computeHomo(Eigen::Matrix<double,3,3>& currHomo,
                                const vector<pair<KeyPoint,KeyPoint>> pairs)
{
    int n = pairs.size();
    Eigen::MatrixXd A(n*2,9);

    for(int i = 0; i < n; i++)
    {
        const double u1 = pairs[i].first.pt.x;
        const double v1 = pairs[i].first.pt.y;
        const double u2 = pairs[i].second.pt.x;
        const double v2 = pairs[i].second.pt.y;

        A(2*i, 0) = u1;
        A(2*i, 1) = v1;
        A(2*i, 2) = 1;
        A(2*i, 3) = 0;
        A(2*i, 4) = 0;
        A(2*i, 5) = 0;
        A(2*i, 6) = -u1 * u2;
        A(2*i, 7) = -v1 * u2;
        A(2*i, 8) = -u2;

        A(2*i+1, 0) = 0;
        A(2*i+1, 1) = 0;
        A(2*i+1, 2) = 0;
        A(2*i+1, 3) = u1;
        A(2*i+1, 4) = v1;
        A(2*i+1, 5) = 1;
        A(2*i+1, 6) = -u1 * v2;
        A(2*i+1, 7) = -v1 * v2;
        A(2*i+1, 8) = -v2;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix<double,9,9> v = svd.matrixV();

    currHomo(0,0) = v(0,8); currHomo(0,1) = v(1,8); currHomo(0,2) = v(2,8);
    currHomo(1,0) = v(3,8); currHomo(1,1) = v(4,8); currHomo(1,2) = v(5,8);
    currHomo(2,0) = v(6,8); currHomo(2,1) = v(7,8); currHomo(2,2) = v(8,8);
}

//@param isInliers 通过bool存储当前匹配是否是inliers
//@param score 判断当前单应矩阵的分数
 void mvg::checkHomo(vector<bool>& isInliers, double& score, Eigen::Matrix<double,3,3>& currHomo)
 {
     Eigen::Matrix<double,3,3> homolr = mSim3Nr.inverse() * currHomo * mSim3Nl;
     Eigen::Matrix<double,3,3> homorl = homolr.inverse();
     
     double hlr0 = homolr(0,0), hlr1 = homolr(0,1), hlr2 = homolr(0,2),
            hlr3 = homolr(1,0), hlr4 = homolr(1,1), hlr5 = homolr(1,2),
            hlr6 = homolr(2,0), hlr7 = homolr(2,1), hlr8 = homolr(2,2);
     double hrl0 = homorl(0,0), hrl1 = homorl(0,1), hrl2 = homorl(0,2),
            hrl3 = homorl(1,0), hrl4 = homorl(1,1), hrl5 = homorl(1,2),
            hrl6 = homorl(2,0), hrl7 = homorl(2,1), hrl8 = homorl(2,2);

     score = 0;
     int n = mMatches.size();
     double reprojectionThresh = 12;

     for(int i = 0; i < n; i++)
     {
         bool bin = true;
         
         double u1 = mKeypointl_r[i].first.pt.x;
         double v1 = mKeypointl_r[i].first.pt.y;
         double u2 = mKeypointl_r[i].second.pt.x;
         double v2 = mKeypointl_r[i].second.pt.y;

         double u2lr = (hlr0 * u1 + hlr1 * v1 + hlr2) / (hlr6 * u1 + hlr7 * v1 + hlr8);
         double v2lr = (hlr3 * u1 + hlr4 * v1 + hlr5) / (hlr6 * u1 + hlr7 * v1 + hlr8);

         double squrelr = (u2 - u2lr) * (u2 - u2lr) + (v2 - v2lr) * (v2 - v2lr);

         if(squrelr > reprojectionThresh)
            bin = false;
         else
            score += reprojectionThresh - squrelr;

         double u1rl = (hrl0 * u2 + hrl1 * v2 + hrl2) / (hrl6 * u2 + hrl7 * v2 + hrl8);
         double v1rl = (hrl3 * u2 + hrl4 * v2 + hrl5) / (hrl6 * u2 + hrl7 * v2 + hrl8);

         double squrerl = (u1 - u1rl) * (u1 - u1rl) + (v1 - v1rl) * (v1 - v1rl);

         if(squrerl > reprojectionThresh)
            bin = false;
         else
            score += reprojectionThresh - squrerl;

         if(bin)
            isInliers[i] = true;
         else
            isInliers[i] = false;

     }
 }

//@param maxInterations ransac进行的最大迭代次数，在这么多次数中选出score最大即可
void mvg::ransacAffine(const int maxIterations)
{
    /*首先需要一个随机数生成器，生成计算模型所需的最少随机数量 */
    mt19937 random_generator(mt19937::default_seed);
    vector<vector<size_t>> randomIndexGroup; // 生成最大迭代组数的随机数组
    vector<size_t> vec_index(mMatches.size());
    std::iota(vec_index.begin(),vec_index.end(),0);
    
    for(int i = 0; i < maxIterations; i++)
    {
		vector<size_t> randomIndexOneGroup(8,0);
        UniformSample<size_t,mt19937,uint32_t>(size_t(8), random_generator,
                                                &vec_index, &randomIndexOneGroup);
        randomIndexGroup.push_back(randomIndexOneGroup);
    }

    vector<bool> isInliersCurr(mMatches.size(), false);
    double score = 0;
    double currScore = 0;
    /*根据最大次数进行迭代，从中取出分数最大的即可*/
    for(int i = 0; i < maxIterations; i++)
    {
        vector<pair<KeyPoint, KeyPoint>> selectedPairs(8);
        Eigen::Matrix<double,3,3> currHomo;
        for(int j = 0; j < 8; j++)
        {
            selectedPairs[j] = mKeypointl_r_norm[randomIndexGroup[i][j]];
        }
        //计算当前八个点的单应矩阵
        computeHomo(currHomo, selectedPairs);
        //判断当前单应矩阵下的inliers
        checkHomo(isInliersCurr, currScore, currHomo);

        if(currScore > score)
        {
			score = currScore;
            mIsInliers_Homo = isInliersCurr;
            mHomoMat = mSim3Nr.inverse() * currHomo * mSim3Nr;
        }
    }
}

//通过ransac去除outlier后，进行最小二乘求解单应矩阵
void mvg::computeHomo_leastSquare()
{
    vector<pair<KeyPoint, KeyPoint>> inlierPairs;
    for(int i = 0; i < mMatches.size(); i++)
    {
        if(mIsInliers_Homo[i])
            inlierPairs.push_back(mKeypointl_r_norm[i]);
    }
    Eigen::Matrix<double,3,3> currHomo;
    computeHomo(currHomo, inlierPairs);

    mHomoMat = mSim3Nr.inverse() * currHomo * mSim3Nl;
}

/*
@param fundaMat 基础矩阵
@param paris 用于计算基础矩阵的像素点对 */
//8点法计算基础矩阵
//1。首先是线性解法，求解Ax=0,解出一个初始的f
//2. 增加一个奇异性约束，将初始的f进行奇异值分解，令最小的奇异值为0
void mvg::computeFundamental(Eigen::Matrix<double,3,3>& fundaMat,
                            vector<pair<KeyPoint,KeyPoint>> pairs)
{
    const size_t n = pairs.size();
    Eigen::MatrixXd A(n,9);

    for(int i = 0; i<n; i++)
    {
        size_t u1 = pairs[i].first.pt.x;
        size_t v1 = pairs[i].first.pt.y;
        size_t u2 = pairs[i].second.pt.x;
        size_t v2 = pairs[i].second.pt.y;

        A(i,0) = u1*u2;
		A(i,1) = v1*u2;
		A(i,2) = u2;
        A(i,3) = u1*v2;
		A(i,4) = v1*v2;
		A(i,5) = v2;
        A(i,6) = u1;
		A(i,7) = v1;
		A(i,8) = 1;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix<double,9,9> v = svd.matrixV();

    fundaMat(0,0) = v(0,8); fundaMat(0,1) = v(1,8); fundaMat(0,2) = v(2,8);
    fundaMat(1,0) = v(3,8); fundaMat(1,1) = v(4,8); fundaMat(1,2) = v(5,8);
    fundaMat(2,0) = v(6,8); fundaMat(2,1) = v(7,8); fundaMat(2,2) = v(8,8);

	Eigen::JacobiSVD<Eigen::MatrixXd> svd_f(fundaMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<double, 3, 3> u_f = svd_f.matrixU();
	Eigen::Matrix<double, 3, 3> v_f = svd_f.matrixV();
	Eigen::Matrix<double, 3, 1> w_f = svd_f.singularValues();
	
	Eigen::Matrix<double, 3, 3> w_m = Eigen::Matrix<double, 3, 3>::Zero();
	w_m(0, 0) = w_f(0, 0);	w_m(1, 1) = w_f(1, 0);
	fundaMat = u_f * w_m * v_f.transpose();
}

/*基础矩阵似乎不用再乘以相似变换矩阵，因为f矩阵的奇异性？*/
/*@param isInliers 存储在基础矩阵模型中ransac的inliers
@param score 通过重投影误差来统计该基础矩阵的分数
@param currFund 当前ransac计算得到的基础矩阵 */
void mvg::checkFunda(vector<bool>& isInliers, double& score, Eigen::Matrix<double,3,3>& currFund)
{
    const size_t n = isInliers.size();

    const double f11 = currFund(0,0), f12 = currFund(0,1), f13 = currFund(0,2),
                 f21 = currFund(1,0), f22 = currFund(1,1), f23 = currFund(1,2),
                 f31 = currFund(2,0), f32 = currFund(2,1), f33 = currFund(2,2);

    bool bIn = false;
    score = 0;
    const double dist_thresh = 5;

    for(int i = 0; i<n; i++)
    {
        const float u1 = mKeypointl_r[i].first.pt.x;
        const float v1 = mKeypointl_r[i].first.pt.y;
        const float u2 = mKeypointl_r[i].second.pt.x;
        const float v2 = mKeypointl_r[i].second.pt.y;          

        /*计算(u1,v1)经过基础矩阵投影到对应图像上的极线方程 */
        double a2 = u1*f11 + v1*f12 + f13;
        double b2 = u1*f21 + v1*f22 + f23;
        double c2 = u1*f31 + v1*f32 + f33;

        double epipolar_norm1 = u2*a2 + v2*b2 + c2;
        const double distance1 = sqrtf(epipolar_norm1 * epipolar_norm1 / (a2 * a2 + b2 * b2)); 
		if (distance1 < dist_thresh)
		{
			bIn = true;
			score += dist_thresh - distance1;
		}
		else
		{
			bIn = false;
		}

		double a1 = u2 * f11 + v2 * f21 + f31;
		double b1 = u2*f12 + v2 * f22 + f32;
		double c1 = u2*f13 + v2 * f23 + f33;

        /*计算对应点到极线的距离，作为判断*/
		double epipolar_norm2;
		epipolar_norm2  = u1 * a1 + v1 * b1 + c1;
        double distance2 = epipolar_norm2 * epipolar_norm2 / (a1 * a1 + b1 * b1);
		if (distance2 < dist_thresh)
		{
			bIn = true;
			score += dist_thresh - distance2;
		}
		else
		{
			bIn = false;
		}

		if (bIn)
		{
			isInliers[i] = true;
		}
		else
		{
			isInliers[i] = false;
		}
    }
}

void mvg::showEpipolarLine(const Mat& src1, const Mat& src2)
{
    Mat dst1,dst2;
    dst1.create(src1.rows, src1.cols, src1.type());
    dst2.create(src2.rows, src2.cols, src2.type());

    src1.copyTo(dst1);
    src2.copyTo(dst2);

    size_t n = mIsInliers_Funda.size();
#pragma omp parallel for
    for(int i = 0; i<n; i++)
    {
        if(mIsInliers_Funda[i])
        {
            const float u1 = mKeypointl_r[i].first.pt.x;
            const float v1 = mKeypointl_r[i].first.pt.y;
            const float u2 = mKeypointl_r[i].second.pt.x;
            const float v2 = mKeypointl_r[i].second.pt.y;

            const double f11 = mFundaMat(0,0);
            const double f12 = mFundaMat(0,1);
            const double f13 = mFundaMat(0,2);
            const double f21 = mFundaMat(1,0);
            const double f22 = mFundaMat(1,1);
            const double f23 = mFundaMat(1,2);
            const double f31 = mFundaMat(2,0);
            const double f32 = mFundaMat(2,1);
            const double f33 = mFundaMat(2,2);

            double img2LineA = f11 * u1 + f12 * v1 + f13;
            double img2LineB = f21 * u1 + f22 * v1 + f23;
            double img2LineC = f31 * u1 + f32 * v1 + f33;

            double img1LineA = f11 * u2 + f21 * v2 + f31;
            double img1LineB = f12 * u2 + f22 * v2 + f32;
            double img1LineC = f13 * u2 + f23 * v2 + f33;

            Point2f start1(0,0), end1(src1.cols,0),start2(0,0), end2(src2.cols,0);
            start1.y = -(img1LineA * start1.x + img1LineC) / img1LineB;
            end1.y = -(img1LineA * end1.x + img1LineC) / img1LineB;
            start2.y = -(img2LineA * start2.x + img2LineC) / img2LineB;
            end2.y = -(img2LineA * end2.x + img2LineC) / img2LineB;

            Scalar color(255.*rand()/RAND_MAX, 255.*rand()/RAND_MAX, 255.*rand()/RAND_MAX);
			circle(dst1, mKeypointl_r[i].first.pt, 1, color, cv::LINE_AA);
			circle(dst2, mKeypointl_r[i].second.pt, 1, color, LINE_AA);
            line(dst1,start1,end1,color);
            line(dst2,start2,end2,color);
        }
    }
	namedWindow("epipolar1", WINDOW_NORMAL);
	namedWindow("epipolar2", WINDOW_NORMAL);

	imshow("epipolar1", dst1);
	imshow("epipolar2", dst2);

    imwrite("epipolarLine1.png",dst1);
    imwrite("epipolarLine2.png",dst2);
}

//通过ransac去除基础矩阵的outlier
//@param maxIterations ransac要进行的最大迭代次数
//通过ransac去除inliers，并计算基础矩阵
void mvg::ransacFundamental(const int maxIterations)
{
    /*首先需要一个随机数生成器，生成计算模型所需的最少随机数量 */
    mt19937 random_generator(mt19937::default_seed);
    vector<vector<size_t>> randomIndexGroup; // 生成最大迭代组数的随机数组
    vector<size_t> vec_index(mMatches.size());
    std::iota(vec_index.begin(),vec_index.end(),0);
    
    for(int i = 0; i < maxIterations; i++)
    {
		vector<size_t> randomIndexOneGroup(8,0);
        UniformSample<size_t,mt19937,uint32_t>(size_t(8), random_generator,
                                                &vec_index, &randomIndexOneGroup);
        randomIndexGroup.push_back(randomIndexOneGroup);
    }

    vector<bool> isInliersCurr(mMatches.size(), false);
    double score = 0;
    double currScore = 0;
    /*根据最大次数进行迭代，从中取出分数最大的即可*/
    for(int i = 0; i < maxIterations; i++)
    {
        vector<pair<KeyPoint,KeyPoint>> selectedPairs;
        Eigen::Matrix<double,3,3> currFundamentalMat;
        for(int j = 0; j < 8; j++)
        {
            selectedPairs.push_back(mKeypointl_r[randomIndexGroup[i][j]]);
        }

        computeFundamental(currFundamentalMat,selectedPairs);
        checkFunda(isInliersCurr,currScore,currFundamentalMat);

        if(currScore > score)
        {
            score = currScore;
            mIsInliers_Funda = isInliersCurr;
            mFundaMat = currFundamentalMat;
        }
		selectedPairs.clear();
    }

	showEpipolarLine(mImgl, mImgr);
}

//从经过ransac之后的匹配对，恢复基础矩阵
void mvg::computeFunda_leastSquare()
{
     vector<pair<KeyPoint,KeyPoint>> inliers_pair;
     for(int i = 0; i<mKeypointl_r.size(); i++)
     {
         if(mIsInliers_Funda[i])
         {
             inliers_pair.push_back(mKeypointl_r[i]);
         }
     }

     computeFundamental(mFundaMat,inliers_pair);
}

/*[-1,0,u;0,-1,v] * projectiveMat * p_3d = 0
利用svd分解求解齐次线性方程组，即可完成三角化 */
/*
@param p1 左图关键点
@param p2 右图关键点
@param R1 全局坐标系转换到左相机下的旋转矩阵
@param R2 全局坐标系转换到右相机下的旋转矩阵
@param t1 全局坐标系平移到左相机下的平移向量
@param t2 全局坐标系平移到右相机下的平移向量 
*/
Eigen::Matrix<double,3,1> mvg::triangulate(const KeyPoint& p1, const KeyPoint& p2,
                            const Eigen::Matrix<double,3,3>& R1,
                            const Eigen::Matrix<double,3,3>& R2,
                            const Eigen::Matrix<double,3,1>& t1,
                            const Eigen::Matrix<double,3,1>& t2)
{
    Eigen::Matrix<double,3,1> p_3d;

    //构建投影矩阵
    Eigen::Matrix<double,3,4> project1, project2;
    project1(0,0) = R1(0,0); project1(0,1) = R1(0,1); project1(0,2) = R1(0,2); project1(0,3) = t1(0,0);
    project1(1,0) = R1(1,0); project1(1,1) = R1(1,1); project1(1,2) = R1(1,2); project1(1,3) = t1(1,0);
    project1(2,0) = R1(2,0); project1(2,1) = R1(2,1); project1(2,2) = R1(2,2); project1(2,3) = t1(2,0);

    project2(0,0) = R2(0,0); project2(0,1) = R2(0,1); project2(0,2) = R2(0,2); project2(0,3) = t2(0,0);
    project2(1,0) = R2(1,0); project2(1,1) = R2(1,1); project2(1,2) = R2(1,2); project2(1,3) = t2(1,0);
    project2(2,0) = R2(2,0); project2(2,1) = R2(2,1); project2(2,2) = R2(2,2); project2(2,3) = t2(2,0); 

    project1 = mIntrinsicK * project1;
    project2 = mIntrinsicK * project2;

    Eigen::Matrix<double,2,3> _p1,_p2;
    _p1<<-1,0,p1.pt.x,0,-1,p1.pt.y;
    _p2<<-1,0,p2.pt.x,0,-1,p2.pt.y;

    Eigen::Matrix<double,2,4> a1 = _p1 * project1;
    Eigen::Matrix<double,2,4> a2 = _p2 * project2;

    //构建系数矩阵
    Eigen::Matrix<double,4,4> A;
    A(0,0) = a1(0,0); A(0,1) = a1(0,1); A(0,2) = a1(0,2); A(0,3) = a1(0,3);
    A(1,0) = a1(1,0); A(1,1) = a1(1,1); A(1,2) = a1(1,2); A(1,3) = a1(1,3);
    A(2,0) = a2(0,0); A(2,1) = a2(0,1); A(2,2) = a2(0,2); A(2,3) = a2(0,3);
    A(3,0) = a2(1,0); A(3,1) = a2(1,1); A(3,2) = a2(1,2); A(3,3) = a2(1,3);

    //利用SVD分解求解Ax=0
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix<double,4,4> v = svd.matrixV(); 

    //因为三维点的表示是四维的齐次坐标，因此要除以第四个元素
    if(v(3,3) != 0)
    {
        p_3d(0,0) = v(0,3) / v(3,3);
        p_3d(1,0) = v(1,3) / v(3,3);
        p_3d(2,0) = v(2,3) / v(3,3);
    }

    return p_3d;
}

/*@param R旋转矩阵
 @param t平移向量
 @param vgood 存储匹配中能够满足条件，生成三维点的bool
 @param th 用于重投影误差判断的阈值
 @param p3d 存储生成的三维点云 */
void mvg::checkRT(const Eigen::Matrix<double,3,3>& R, const Eigen::Matrix<double,3,1>& t,
                vector<bool>& vgood, const float th, vector<Eigen::Vector3d>& p3d,
                size_t& nGood)
{
    const size_t n = mMatches.size();
    vgood.resize(n);
    p3d.resize(n);
    nGood = 0;

    Eigen::Matrix<double,3,1> o1 = Eigen::Matrix<double,3,1>::Zero();
    Eigen::Matrix<double,3,1> o2 = -R.transpose()*t; //右侧相机光心在左侧相机坐标系（全局坐标系）下的坐标
    for(int i = 0; i<n; i++)
    {
        //判断是否在inliers中,不在inlier中则必然不会参与点云生成
        if(!mIsInliers_Funda[i])
        {
            vgood[i] = false;
            continue;
        }
        const KeyPoint p1 = mKeypointl_r[i].first;
        const KeyPoint p2 = mKeypointl_r[i].second;

        //三角化对应点
        Eigen::Matrix<double,3,1> p3dc1 = triangulate(p1,p2,Eigen::Matrix<double,3,3>::Identity(),
                                            R,Eigen::Matrix<double,3,1>::Zero(),t);

        //三维点与图像对应点间向量的夹角
        Eigen::Matrix<double,3,1> vectorP_o1 = p3dc1 - o1;
        Eigen::Matrix<double,3,1> vectorP_o2 = p3dc1 - o2;

        float cosineVector = vectorP_o1.dot(vectorP_o2) / (vectorP_o1.norm() * vectorP_o2.norm());
        //三维点与对应点间向量的夹角就是视差角，剔除视差角较小的三维点，并且将成像在光心之后的点剔除
        //按道理来说，只有一种情况三维点全部在光心之前，仅靠这个条件也可以判断，但是可能存在误差，导致
        //姿态正确的情况下，匹配错误，三维点可能出现在光心之后，因此还是采用符合条件的点最多的条件来
        //找正确的姿态。
        if(p3dc1(2,0)<0 && cosineVector < 1)
        {
            vgood[i] = false;
            continue;
        }

        //进行重投影，根据阈值重新剔除
        Eigen::Matrix<double,3,1> p3dc2 = R * p3dc1 + t;
        float reprojectp1_u = 1/p3dc1(2,0) * (mIntrinsicK(0,0)*p3dc1(0,0) + mIntrinsicK(0,2));
        float reprojectp1_v = 1/p3dc1(2,0) * (mIntrinsicK(1,1)*p3dc1(1,0) + mIntrinsicK(1,2));
        float reprojectp2_u = 1/p3dc2(2,0) * (mIntrinsicK(0,0)*p3dc2(0,0) + mIntrinsicK(0,2));
        float reprojectp2_v = 1/p3dc2(2,0) * (mIntrinsicK(1,1)*p3dc2(1,0) + mIntrinsicK(1,2));
        //计算重投影误差
        float err1 = sqrtf((reprojectp1_u - p1.pt.x) * (reprojectp1_u - p1.pt.x) +
                            (reprojectp1_v - p1.pt.y) * (reprojectp1_v - p1.pt.y));
        float err2 = sqrtf((reprojectp2_u - p2.pt.x) * (reprojectp2_u - p2.pt.x) +
                            (reprojectp2_v - p2.pt.y) * (reprojectp2_v - p2.pt.y));

        if(err1 > th || err2 > th)
        {
            vgood[i] = false;
            continue;
        }
        nGood++;
    }
}

/*从f矩阵恢复R,T和三维坐标，
因为f矩阵的秩为2，则f的奇异值为[sita,sita,0];又因为e=t^R,
t^是一个反对称矩阵，R是一个正交矩阵，因此
反对称矩阵可以写成 z=[0,1,0;-1,0,0;0,0,0]*/
void mvg::reconstructF()
{
    ///*1.首先通过最小二乘从ransac挑选出的inliers里进行拟合*/

    computeFunda_leastSquare();   
    mEssenMat = mIntrinsicK.transpose() * mFundaMat * mIntrinsicK;

    /*2.利用得到的本质矩阵分解得到R,T
    z = [0 1 0; -1 0 0; 0 0 0];
    w = [0 -1 0; 1 0 0; 0 0 1];
    z*w = [1 0 0; 0 1 0; 0 0 0];
    z*wt = -[1 0 0; 0 1 0; 0 0 0];
    f = UDVt = UzwVt = UzUt* UwVt = t^ * R;
    f = UDVt = -UzwtVt = -UzUt*UwtVt = t^ * R;
    所以:
    t^ = UzUt; t^ = -UzUt; 
    R = UwVt; R = UwtVt;
    因此，从基础矩阵会分解出四组解，但是只有一组解在相机前方。
    */

   Eigen::JacobiSVD<Eigen::MatrixXd> svd(mEssenMat, Eigen::ComputeThinU || Eigen::ComputeThinV);
   Eigen::Matrix<double, 3, 3> V_rt = svd.matrixV();
   Eigen::Matrix<double, 3, 3> U_rt = svd.matrixU();
    
   Eigen::Matrix<double,3,3> z,w;
   z<<0,1,0,-1,0,0,0,0,0;
   w<<0,-1,0,1,0,0,0,0,1;

   Eigen::Matrix<double,3,3> R1 = U_rt * w * V_rt.transpose();
   Eigen::Matrix<double,3,3> R2 = U_rt * w.transpose() * V_rt.transpose();

   //旋转矩阵的属性，旋转矩阵的行列式大于0
   if(R1.determinant() < 0)
        R1 = -R1;
   if(R2.determinant() < 0)
        R2 = -R2;
    
    Eigen::Matrix<double,3,3>t1_cross = U_rt * z * U_rt.transpose();
    Eigen::Matrix<double,3,1>t1,t2;
    t1(0,0) = -t1_cross(1,2);
    t1(1,0) = t1_cross(0,2);
    t1(2,0) = -t1_cross(0,1);
    t2 = -t1; 

    //检查四组姿态中的合格姿态，并将其进行三维点云生成
    vector<bool> vgood1,vgood2,vgood3,vgood4;
    size_t ngood1,ngood2,ngood3,ngood4;
    vector<Eigen::Vector3d> p3d1,p3d2,p3d3,p3d4;
    const float th = 5;

    //检查svd分解得到的r，t，并进行三角化，选出符合要求的structure和motion
    checkRT(R1,t1,vgood1,th,p3d1,ngood1);
    checkRT(R1,t2,vgood2,th,p3d2,ngood2);
    checkRT(R2,t1,vgood3,th,p3d3,ngood3);
    checkRT(R2,t2,vgood4,th,p3d4,ngood4);

    vector<size_t> goodNum{ngood1, ngood2, ngood3, ngood4};
    size_t maxGood = max<size_t>(goodNum);

    mP3d.resize(mKeypointl_r.size());
    mIsTriangulate.resize(mKeypointl_r.size());

    if(ngood1 == maxGood)
    {
        mIsTriangulate = vgood1;
        mP3d = p3d1;
    }
    else if(ngood2 == maxGood)
    {
        mIsTriangulate = vgood2;
        mP3d = p3d2;
    }
    else if(ngood3 == maxGood)
    {
        mIsTriangulate = vgood3;
        mP3d = p3d3;
    }
    else if(ngood4 == maxGood)
    {
        mIsTriangulate = vgood4;
        mP3d = p3d4;
    }
}