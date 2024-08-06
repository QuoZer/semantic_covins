#ifndef WALLSEG_HPP_
#define WALLSEG_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


using namespace cv;
using namespace std;


class CloudWallSegmentation{
  public:
    pcl::PointCloud<pcl::PointXYZRGB> processPointcloud(pcl::PointCloud<pcl::PointXYZRGB> inputCloud);

    Mat createProjectionImage(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, int outWidth, int outHeight);

    Mat DrawTrajectories (vector<vector<float>> trajectories, Mat cloudProjection);

    Mat DrawTrajectoriesD (vector<vector<double>> trajectories, Mat cloudProjection);

    // Mat DrawTrajectoriesTime (vector<vector<float>> trajectories, Mat cloudProjection, int timeSinceStart, int timeLimit);

    Mat DrawTrajectoriesTime (vector<vector<float>> trajectories, Mat cloudProjection, vector<int> timeSinceStart, int timeLimit);

    Mat DrawTrajectoriesTime2(vector<float> trajectories, vector<int> sizes,Mat cloudProjection, vector<int> timeSinceStart, int timeLimit);

  private:
    float minThresh = 0;
    float maxThresh = 0;
    float minX = 0;
    float minY = 0;
    float resolution = 1;
    int shiftH = 0;
    int shiftW = 0;
    Scalar trajColors[5] = {Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(255,0,255)};
    // vector<int> timeStamps;
    // vector<int> trajSize;
    vector<vector<int>> timeStamps;
    vector<vector<int>> trajSize;


    pcl::PointCloud<pcl::PointXYZRGB> MlsProcessing(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, int polyOrder, double searchRadius);

    pcl::PointCloud<pcl::PointXYZRGB> wallSegmentation(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, double epsAngle, double distThresh, double pointCountThresh);

    pcl::PointCloud<pcl::PointXYZRGB> removeOutliers(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, int meanK, double thresh);

    Mat projectionImage(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, float resolution);
    
    Mat projectionImageRes(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, int outWidth, int outHeight);

    pcl::PointCloud<pcl::PointXYZRGB> DistributionCut(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, float valueWindow);

    pcl::PointCloud<pcl::PointXYZRGB> DistribFastCut(pcl::PointCloud<pcl::PointXYZRGB> inputCloud);

    // int clearTrajectoryOlderThan(int sec);

    vector<int> clearTrajectoryOlderThan(int sec);

};

#endif
