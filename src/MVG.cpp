#include "MVG.hpp"

//@param dir1,dir2 输入图像的地址
//@param nfeatures 检测到图像特征点的数量
//该构造函数用来计算仿射变换
mvg::mvg(string dir1, string dir2, const int nfeatures, const float ratio,
        const int maxIterations)
{
    mImgl = imread(dir1);
    mImgr = imread(dir2);

    computeFeat(nfeatures);
    showKeyPoint();

    computeMatches(ratio);
    showMatch();

	if (mMatches.size() < 100)
		cerr << "Attention!!! There is no enough matches to compute affine matrix!!!\n";

    normalize();
    ransacAffine(maxIterations);
    computeHomo_leastSquare();
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
        cout<<"maybe there is no enough keypoints\n";
    }

    Mat keypointImgl,keypointImgr;
    drawKeypoints(mImgl,mKeypointl,keypointImgl,Scalar(255,0,0));
    drawKeypoints(mImgr,mKeypointr,keypointImgr,Scalar(255,0,0));

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
        cout << "Maybe there is no enough match\n";
    }
    Mat imgMatch;
    drawMatches(mImgl, mKeypointl, mImgr, mKeypointr,
                mMatches, imgMatch, Scalar(255,0,0));
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

        A(i,0) = u1*u2; A(i,1) = u1*v2; A(i,2) = u1;
        A(i,3) = v1*u2; A(i,4) = v1*v2; A(i,5) = v1;
        A(i,6) = u2; A(i,7) = v2; A(i,8) = 1;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix<double,9,9> v = svd.matrixV();

    fundaMat(0,0) = v(0,8); fundaMat(0,1) = v(1,8); fundaMat(0,2) = v(2,8);
    fundaMat(1,0) = v(3,8); fundaMat(1,1) = v(4,8); fundaMat(1,2) = v(5,8);
    fundaMat(2,0) = v(6,8); fundaMat(2,1) = v(7,8); fundaMat(2,2) = v(8,8);
}

/*基础矩阵似乎不用再乘以相似变换矩阵，因为f矩阵的奇异性？*/
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
        const int u1 = mKeypointl_r[i].first.pt.x;
        const int v1 = mKeypointl_r[i].first.pt.y;
        const int u2 = mKeypointl_r[i].second.pt.x;
        const int v2 = mKeypointl_r[i].second.pt.y;          

        /*计算(u1,v1)经过基础矩阵投影到对应图像上的极线方程 */
        const int a2 = u1*f11 + v1*f12 + f13;
        const int b2 = u1*f21 + v1*f22 + f23;
        const int c2 = u1*f31 + v1*f32 + f33;

        /*计算对应点到极线的距离，作为判断*/
        const double epipolar_norm1 = u2*a2 + v2*b2 + c2;
        const double distance1 = epipolar_norm1 * epipolar_norm1 / (a2 * a2 + b2 * b2); 
        if(distance1 > dist_thresh)
            bIn = false;
        else
            score += dist_thresh - distance1;

        /*计算(u2,v2)经过基础矩阵投影到对应图像上的极线方程*/
        const int a1 = u2*f11 + v2*f12 + f13;
        const int b1 = u2*f21 + v2*f22 + f23;
        const int c1 = u2*f31 + v2*f32 + f33;

        /*计算对应点到极线的距离，作为判断*/
        const double epipolar_norm2 = u1*a1 + v1*b1 + c1;
        const double distance2 = epipolar_norm2 * epipolar_norm2 / (a1 * a1 + b1 * b1);
        if(distance2 > dist_thresh)
            bIn = true;
        else
            score += dist_thresh - distance2;

        if(bIn)
            isInliers[i] = true;
        else
            isInliers[i] = false;
        
    }
}

//通过ransac去除基础矩阵的outlier
//@param maxIterations ransac要进行的最大迭代次数
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
        vector<pair<KeyPoint,KeyPoint>> selectedPairs(8);
        Eigen::Matrix<double,3,3> currFundamentalMat;
        for(int j = 0; j < 8; j++)
        {
            selectedPairs.push_back(mKeypointl_r_norm[randomIndexGroup[i][j]]);
        }

        computeFundamental(currFundamentalMat,selectedPairs);
        checkFunda(isInliersCurr,currScore,currFundamentalMat);

        if(currScore > score)
        {
            score = currScore;
            mIsInliers_Funda = isInliersCurr;
            mFundaMat = currFundamentalMat;
        }
    }
}

/*从f矩阵恢复R,T和三维坐标*/
void mvg::constructF()
{
    int n = 0;
    for(int i = 0; i < mIsInliers_Funda.size(); i++)
    {
        if(mIsInliers_Funda[i])
            n++;
    }
    /*1.首先通过最小二乘从ransac挑选出的inliers里进行拟合*/
    Eigen::MatrixXd A(n,9);
    for(int i = 0; i < mIsInliers_Funda.size(); i++)
    {
        if(mIsInliers_Funda[i])
        {
            const size_t u1 = mKeypointl_r_norm[i].first.pt.x;
            const size_t v1 = mKeypointl_r_norm[i].first.pt.y;
            const size_t u2 = mKeypointl_r_norm[i].second.pt.x;
            const size_t v2 = mKeypointl_r_norm[i].second.pt.y;

            A(i,0) = u1*u2; A(i,1) = u1*v2; A(i,2) = u1;
            A(i,3) = v1*u2; A(i,4) = v1*v2; A(i,5) = v1;
            A(i,6) = u2;    A(i,7) = v2;    A(i,8) = 1;
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_f(A,Eigen::ComputeFullU||Eigen::ComputeFullV);
    Eigen::Matrix<double,9,9> V = svd_f.matrixV();

    mFundaMat(0,0) = V(0,8); mFundaMat(0,1) = V(1,8); mFundaMat(0,2) = V(2,8);
    mFundaMat(1,0) = V(3,8); mFundaMat(1,1) = V(4,8); mFundaMat(1,2) = V(5,8);
    mFundaMat(2,0) = V(6,8); mFundaMat(2,1) = V(7,8); mFundaMat(2,2) = V(8,8);

    /*2.利用得到的基础矩阵分解得到R,T
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
   Eigen::JacobiSVD<Eigen::MatrixXd> svd_rt(mFundaMat,Eigen::ComputeFullU||Eigen::ComputeFullV);
   Eigen::Matrix<double,3,3> U_rt = svd_rt.matrixU();
   Eigen::Matrix<double,3,3> V_rt = svd_rt.matrixV();


}