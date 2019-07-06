#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

/*Fisher-Yates_shuffle 洗牌算法
  不断缩小范围生成随机数，并且与当前id进行交换，实现随机交换，本质上实现了n！的排列数量*/
template<typename T, class RandomGenerator,
         typename SamplingType = uint32_t>
bool UniformSample
(
    const size_t num_samples,
    RandomGenerator & random_generator,
    vector<T>* vec_index, // 待采样数据的index，应该是随机排列的
    vector<T>* samples // 采样得到的index
)
{
    static_assert(is_integral<SamplingType>::value, "samplingType must be a integral tyep");

    // static_cast就是用来做强制类型转换的
    if(num_samples > vec_index->size() ||
        vec_index->size() > static_cast<size_t>(numeric_limits<SamplingType>::max()))
        return false;

    const SamplingType last_idx(vec_index->size() - 1);
    for(SamplingType i = 0; i < num_samples; ++i)
    {
        std::uniform_int_distribution<SamplingType> distribution(i,last_idx);
        const SamplingType sample = distribution(random_generator);

        std::swap((*vec_index)[i],(*vec_index)[sample]);
    }
    samples->resize(num_samples);
    for(size_t i = 0; i < num_samples; i++)
    {
        (*samples)[i] = (*vec_index)[i];
    }
    return true;
}

class mvg
{
public:
    mvg(string dir1, string dir2, const int nfeatures, const float ratio, const int
        maxIterations);
    void computeFeat(const int nfeatures); //计算特征点和描述子
    void computeMatches(const float ratio); //计算特征点间的匹配
    void ransacAffine(const int maxIterations); //对affine变化利用ransac去除outlier
    void normalize(); //归一化特征点坐标
    void computeHomo(Eigen::Matrix<double,3,3>& currHomo, 
                                vector<pair<KeyPoint,KeyPoint>> pairs);
    void computeHomo_leastSquare();
    void checkHomo(vector<bool>& isInliers, double& score, Eigen::Matrix<double,3,3>& currHomo);
    void showKeyPoint();
    void showMatch();
private:
    Mat mImgl; // 图像
    Mat mImgr; 
    Mat mDescl; // 描述子
    Mat mDescr;

    vector<KeyPoint> mKeypointl; //特征点坐标
    vector<KeyPoint> mKeypointr;
    vector<DMatch> mMatches; //匹配index
    vector<pair<KeyPoint, KeyPoint>> mKeypointl_r; //匹配坐标对
    vector<pair<KeyPoint, KeyPoint>> mKeypointl_r_norm; // 归一化匹配坐标对
    vector<bool> mIsInliers_Homo;

    Eigen::Matrix<double,3,3> mSim3Nl;
    Eigen::Matrix<double,3,3> mSim3Nr;
    Eigen::Matrix<double,3,3> mHomoMat;
};
