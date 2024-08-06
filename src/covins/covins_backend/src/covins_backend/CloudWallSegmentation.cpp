#include <iostream>
#include <vector>
#include <chrono>

#include <pcl/point_cloud.h>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
// #include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>



#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

#include "covins_backend/CloudWallSegmentation.hpp"


pcl::PointCloud<pcl::PointXYZRGB> CloudWallSegmentation::processPointcloud(pcl::PointCloud<pcl::PointXYZRGB> inputCloud) {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  pcl::PointCloud<pcl::PointXYZRGB> outputCloud;

  auto t1 = high_resolution_clock::now();
  // outputCloud = MlsProcessing(inputCloud, 2, 0.5);
  outputCloud = removeOutliers(inputCloud, 30, 0.5);
  // if (this->minThresh == 0 && this->maxThresh == 0) outputCloud = DistributionCut(outputCloud);
  // else outputCloud = DistribFastCut(outputCloud);
  auto t2 = high_resolution_clock::now();
  outputCloud = DistributionCut(outputCloud, 0.25);
  // outputCloud = wallSegmentation(outputCloud, 5, 0.3, 0.2);
  auto t3 = high_resolution_clock::now();
  outputCloud = removeOutliers(outputCloud, 30, 0.5);
  auto t4 = high_resolution_clock::now();

  auto ms_int1 = duration_cast<milliseconds>(t2 - t1);
  auto ms_int2 = duration_cast<milliseconds>(t3 - t2);
  auto ms_int3 = duration_cast<milliseconds>(t4 - t3);

  //std::cerr << "Process durations: " << ms_int1.count() << " ms, " << ms_int2.count() << " ms, " << ms_int3.count() << " ms" << '\n';
  return outputCloud;
}

Mat CloudWallSegmentation::createProjectionImage(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, int outWidth, int outHeight) {
  // Mat cloudProjection = projectionImage(inputCloud, resolution);
  Mat cloudProjection = projectionImageRes(inputCloud, outWidth, outHeight);

  return cloudProjection;
}

pcl::PointCloud<pcl::PointXYZRGB> CloudWallSegmentation::MlsProcessing(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, int polyOrder, double searchRadius){
  // Create a KD-Tree
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);

  // Output has the PointNormal type in order to store the normals calculated by MLS
  pcl::PointCloud<pcl::PointXYZRGBNormal> mls_points;

  // Init object (second point type is for the normals, even if unused)
  pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> mls;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputPtr (new pcl::PointCloud<pcl::PointXYZRGB>);
  *inputPtr = inputCloud;

  mls.setComputeNormals (false);

  // Set parameters
  mls.setInputCloud (inputPtr);
  mls.setPolynomialOrder (polyOrder);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (searchRadius);

  // Reconstruct
  mls.process (mls_points);

  pcl::PointCloud<pcl::PointXYZRGB> mlsCloud;

  for (const auto &point : mls_points.points) {
    pcl::PointXYZRGB temp_p;
    temp_p.x = point.x;
    temp_p.y = point.y;
    temp_p.z = point.z;
    // int alphaValue = (int)(point.z / (minmax[5] - minmax[4]) * 255);
    int alphaValue = 255;
    // int alphaValue = (int)((point.z + 5) / 10 * 255);
    // int alphaint = (int)alphaValue;
    temp_p.r = alphaValue;
    temp_p.g = alphaValue;
    temp_p.b = alphaValue;

    mlsCloud.push_back(temp_p);
    // std::cerr << alphaValue << " ";


  }

  return mlsCloud;
}

pcl::PointCloud<pcl::PointXYZRGB> CloudWallSegmentation::wallSegmentation(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, double epsAngle, double distThresh, double pointCountThresh){
  int planePointCount = inputCloud.size() * 0.3;
  int originalPointCount = inputCloud.size();

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr wallCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  *wallCloud = inputCloud;

  while (planePointCount > originalPointCount * pointCountThresh){
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGB>  seg;   // Create the segmentation object
    seg.setAxis(Eigen::Vector3f::UnitZ());      // Set the axis along which we need to search for a model perpendicular to
    seg.setEpsAngle(M_PI/180*epsAngle);           // Set maximum allowed difference between the model normal and the given axis in radians
    seg.setOptimizeCoefficients(true);          // Coefficient refinement is required
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setDistanceThreshold(distThresh);
    seg.setInputCloud(wallCloud);
    seg.segment(*inliers, *coefficients);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloudPlane (new pcl::PointCloud<pcl::PointXYZRGB>);

    // Extract inliers
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(wallCloud);
    extract.setIndices(inliers);
    extract.setNegative(true);             // Extract the inliers
    extract.filter(*pointcloudPlane);
    planePointCount = wallCloud->size() - pointcloudPlane->size();

    // if (planePointCount > originalPointCount * pointCountThresh &&
    // abs(coefficients->values[0]) < 0.1 &&
    // abs(coefficients->values[1]) < 0.1 &&
    // abs(coefficients->values[2]) > 0.9) wallCloud = pointcloudPlane;

    if (planePointCount > originalPointCount * pointCountThresh) wallCloud = pointcloudPlane;


    //cerr << "Number of points in plane: " << planePointCount << endl;
    // std::cerr << "Plane Model coefficients: " << coefficients->values[0] << " "
    // << coefficients->values[1] << " "
    // << coefficients->values[2] << " "
    // << coefficients->values[3] << std::endl;


  }

  return *wallCloud;
}

pcl::PointCloud<pcl::PointXYZRGB> CloudWallSegmentation::removeOutliers(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, int meanK, double thresh){
  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputPtr (new pcl::PointCloud<pcl::PointXYZRGB>);
  *inputPtr = inputCloud;
  sor.setInputCloud (inputPtr);
  sor.setMeanK (meanK);
  sor.setStddevMulThresh (thresh);
  sor.filter (inputCloud);

  // pcl::io::savePCDFileASCII("outfile_sor.pcd", inputCloud);

  return inputCloud;
}

Mat CloudWallSegmentation::projectionImage(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, float resolution) {

  std::array<float, 4> minmax{  std::numeric_limits<float>::max(),  - std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), - std::numeric_limits<float>::max() };
  // std::array<float, 6> minmax{  std::numeric_limits<float>::max(),  - std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), - std::numeric_limits<float>::max() , std::numeric_limits<float>::max(), - std::numeric_limits<float>::max() };
  for (const auto &point : inputCloud.points)
  {
    if (point.x < minmax[0])
    {
      minmax[0] = point.x;
    }

    if (point.x > minmax[1])
    {
      minmax[1] = point.x;
    }

    if (point.y < minmax[2])
    {
      minmax[2] = point.y;
    }

    if (point.y > minmax[3])
    {
      minmax[3] = point.y;
    }
   // if (point.z < minmax[4])
   // {
   //   minmax[4] = point.z;
   // }
   //
   // if (point.z > minmax[5])
   // {
   //   minmax[5] = point.z;
   // }

  }

  // int imHeight = (int)((minmax[1] - minmax[0]) / resolution);
  // int imWidth = (int)((minmax[3] - minmax[2]) / resolution);
  int imHeight = ceil((minmax[1] - minmax[0]) / resolution);
  int imWidth = ceil((minmax[3] - minmax[2]) / resolution);
  // int averageZ = (int)((minmax[5] - minmax[4]) / 2);

  //cerr << "Image height and width: " << imHeight << " * " << imWidth << endl;

  unsigned char buffer[imHeight * imWidth] = { 0 };


  for (const auto &point : inputCloud.points) {
    int pointH = floor((point.x - minmax[0]) / resolution);
    int pointW = floor((point.y - minmax[2]) / resolution);
    // int pointH = 0;
    // int pointW = 0;
    // if (imHeight < imWidth) {
    //   // pointH = (int)((point.x - minmax[0]) / resolution);
    //   // pointW = (int)((point.y - minmax[2]) / resolution);
    //   pointH = floor((point.x - minmax[0]) / resolution);
    //   pointW = floor((point.y - minmax[2]) / resolution);
    // }
    // else {
    //   pointW = (int)((point.x - minmax[0]) / resolution);
    //   pointH = (int)((point.y - minmax[2]) / resolution);
    // }
    // int intensity = (int)((point.z - minmax[4]) / (minmax[5] - minmax[4]) * 255) + averageZ;
    // if (intensity > 255) intensity = 255;
    if(buffer[pointH * imWidth + pointW] < (unsigned char)point.r) {
      // buffer[pointH * imWidth + pointW] = (unsigned char)intensity;
      buffer[pointH * imWidth + pointW] = (unsigned char)255;
    }
  }

  // Mat projection = Mat(imHeight, imWidth, CV_8UC1, buffer);
  this->minX = minmax[0];
  this->minY = minmax[2];
  this->resolution = resolution;

  return Mat(imHeight, imWidth, CV_8UC1, buffer).clone();
}

Mat CloudWallSegmentation::projectionImageRes(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, int outWidth, int outHeight) {

  int origWidth = outWidth;
  int origHeight = outHeight;
  std::array<float, 4> minmax{  std::numeric_limits<float>::max(),  - std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), - std::numeric_limits<float>::max() };
  // std::array<float, 6> minmax{  std::numeric_limits<float>::max(),  - std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), - std::numeric_limits<float>::max() , std::numeric_limits<float>::max(), - std::numeric_limits<float>::max() };
  for (const auto &point : inputCloud.points)
  {
    if (point.x < minmax[0])
    {
      minmax[0] = point.x;
    }

    if (point.x > minmax[1])
    {
      minmax[1] = point.x;
    }

    if (point.y < minmax[2])
    {
      minmax[2] = point.y;
    }

    if (point.y > minmax[3])
    {
      minmax[3] = point.y;
    }
   // if (point.z < minmax[4])
   // {
   //   minmax[4] = point.z;
   // }
   //
   // if (point.z > minmax[5])
   // {
   //   minmax[5] = point.z;
   // }

  }
  outWidth = outWidth - 2;
  outHeight = outHeight - 2;
  // int imHeight = (int)((minmax[1] - minmax[0]) / resolution);
  // int imWidth = (int)((minmax[3] - minmax[2]) / resolution);
  float imHeight = minmax[1] - minmax[0];
  float imWidth = minmax[3] - minmax[2];
  // int averageZ = (int)((minmax[5] - minmax[4]) / 2);
  float heightRatio = imHeight / (float)outHeight;
  float widthRatio = imWidth / (float)outWidth;
  float resolution = 0.1;
  int shiftH = 0;
  int shiftW = 0;

  if (heightRatio > widthRatio){
    resolution = heightRatio;
    shiftW = floor((origWidth - imWidth / heightRatio) / 2);
  } 
  else {
    resolution = widthRatio;
    shiftH = floor((origHeight - imHeight / widthRatio) / 2);
  } 

  //cerr << "Image height and width: " << imHeight << " * " << imWidth << endl;
  //cerr << "Height and width ratios: " << heightRatio << " * " << widthRatio << endl;

  unsigned char buffer[origHeight * origWidth] = { 0 };


  for (const auto &point : inputCloud.points) {
    int pointH = floor((point.x - minmax[0]) / resolution) + shiftH;
    int pointW = floor((point.y - minmax[2]) / resolution) + shiftW;

    if(buffer[pointH * origWidth + pointW] < (unsigned char)point.r) {

      buffer[pointH * origWidth + pointW] = (unsigned char)255;
    }
  }

  this->minX = minmax[0];
  this->minY = minmax[2];
  this->resolution = resolution;
  this->shiftH = shiftH;
  this->shiftW = shiftW;


  return Mat(origHeight, origWidth, CV_8UC1, buffer).clone();
}

pcl::PointCloud<pcl::PointXYZRGB> CloudWallSegmentation::DistributionCut(pcl::PointCloud<pcl::PointXYZRGB> inputCloud, float valueWindow){
    //size of unit along Z
    // float valueWindow = 0.25;
    //find min and max
    float minZ = std::numeric_limits<float>::max();
    float maxZ = - std::numeric_limits<float>::max();
    for (const auto &point : inputCloud.points) {
      if (point.z < minZ)
      {
        minZ = point.z;
      }

      if (point.z > maxZ)
      {
        maxZ = point.z;
      }
    }
    //create distribution
    int windowCount = ceil((maxZ - minZ) / valueWindow);
    std::vector<int> distribVec(windowCount, 0);
    std::vector<float> distribAverage(windowCount, 0);
    for (const auto &point : inputCloud.points) {
      int windowInd = floor((point.z - minZ) / valueWindow);
      distribVec[windowInd]++;
      distribAverage[windowInd]+=point.z;
    }
    //calculate the avarage for each bin
    for (int i=0; i<distribAverage.size();i++){
      distribAverage[i]=distribAverage[i]/distribVec[i];
    }
    //calculate std for each bin
    std::vector<float> distribStd(windowCount, 0);
    for (const auto &point : inputCloud.points) {
      int windowInd = floor((point.z - minZ) / valueWindow);
      distribStd[windowInd]+=std::pow(point.z-distribAverage[windowInd],2);
    }
    for (int i=0; i<distribStd.size();i++){
      distribStd[i]=std::sqrt(distribStd[i]/distribVec[i]);
    }
    //calculate average distribution size
    int averageDistrib = 0;
    std::vector<int> localMax;
    for (int i = 0; i < distribVec.size(); i ++) {
      //std::cerr << "number of points in " << i << " window: " << distribVec[i] << '\n';
      if (distribVec[i] > 0) averageDistrib += distribVec[i];
    }
    averageDistrib = averageDistrib / windowCount;
    //find local maxima
    for (int i = 1; i < distribVec.size() - 1; i ++) {
      if (distribVec[i] > distribVec[i - 1] && distribVec[i] > distribVec[i + 1] && distribVec[i] > averageDistrib)
        localMax.push_back(i);
    }
    pcl::PointCloud<pcl::PointXYZRGB> outputCloud;
    int minThreshIndex = localMax.front();
    int maxThreshIndex = localMax.back();
    //std::cerr << "Min ind: " << minThreshIndex << '\n' << "Max ind: " << maxThreshIndex << '\n';
    int minMargin = 2;
    int maxMargin = 2;
    if (maxThreshIndex - minThreshIndex < minMargin + maxMargin + 2) {
      minThreshIndex = 0;
      maxThreshIndex = distribVec.size() - 1;
      //std::cerr << "Fixed Min ind: " << minThreshIndex << '\n' << "Fixed Max ind: " << maxThreshIndex << '\n';
    }
    // float minThresh = (float)localMax.front() * valueWindow + minZ + valueWindow * 3;
    // float maxThresh = (float)localMax.back() * valueWindow + minZ - valueWindow * 2;
    this->minThresh = (float)minThreshIndex * valueWindow + minZ + valueWindow * minMargin;
    this->maxThresh = (float)maxThreshIndex * valueWindow + minZ - valueWindow * maxMargin;
    //TODO
    this->minThresh = (float)minThreshIndex * valueWindow + minZ + distribStd[minThreshIndex];
    this->maxThresh = (float)maxThreshIndex * valueWindow + minZ - distribStd[maxThreshIndex];
    //std::cout<<"minthresholdmargin: "<<distribStd[minThreshIndex]<<"\n";
    //std::cout<<"maxthresholdmargin: "<<distribStd[maxThreshIndex]<<"\n";
    for (const auto &point : inputCloud.points) {
      if (point.z > this->minThresh && point.z < this->maxThresh) {
        outputCloud.push_back(point);
      }
    }
    if (outputCloud.points.empty()) return inputCloud;

    return outputCloud;
}

pcl::PointCloud<pcl::PointXYZRGB> CloudWallSegmentation::DistribFastCut(pcl::PointCloud<pcl::PointXYZRGB> inputCloud){
  pcl::PointCloud<pcl::PointXYZRGB> outputCloud;
  for (const auto &point : inputCloud.points) {
    if (point.z > minThresh && point.z < maxThresh) {
      outputCloud.push_back(point);
    }
  }
  return outputCloud;
}

Mat CloudWallSegmentation::DrawTrajectories (vector<vector<float>> trajectories, Mat cloudProjection){
  Mat colorProjection;
  cvtColor(cloudProjection, colorProjection, CV_GRAY2RGB);
  for(int j = 0; j < trajectories.size(); j++) {
    for(int i = 0; i < trajectories[j].size() - 4; i += 2) {
      int startY = floor((trajectories[j][i] - this->minX) / this->resolution) + this->shiftH;
      int startX = floor((trajectories[j][i + 1] - this->minY) / this->resolution) + this->shiftW;
      int endY = floor((trajectories[j][i + 2] - this->minX) / this->resolution) + this->shiftH;
      int endX = floor((trajectories[j][i + 3] - this->minY) / this->resolution) + this->shiftW;
      line(colorProjection,Point(startX,startY),Point(endX,endY),this->trajColors[j],2);
    }
  }
  for(int j = 0; j < trajectories.size(); j++) {
    int markerY = floor((trajectories[j][trajectories[j].size() - 2] - this->minX) / this->resolution) + this->shiftH;
    int markerX = floor((trajectories[j][trajectories[j].size() - 1] - this->minY) / this->resolution) + this->shiftW;
    circle(colorProjection,Point(markerX,markerY),12,this->trajColors[j],-1);
  }
  return colorProjection.clone();
}

// Mat CloudWallSegmentation::DrawTrajectoriesTime (vector<vector<float>> trajectories, Mat cloudProjection, int timeSinceStart, int timeLimit){
//   this->timeStamps.push_back(timeSinceStart);
//   this->trajSize.push_back(trajectories[0].size());
//   int startIndex = 0;
//   if (timeStamps.size() > 2) {
//     startIndex = clearTrajectoryOlderThan(timeLimit);
//   }
//   Mat colorProjection;
//   cvtColor(cloudProjection, colorProjection, CV_GRAY2RGB);
//   for(int j = 0; j < trajectories.size(); j++) {
//     // for(int i = 0; i < trajectories[j].size() - 4; i += 2) {
//     while (startIndex < trajectories[j].size() - 4) {
//       int startY = floor((trajectories[j][startIndex] - this->minX) / this->resolution) + this->shiftH;
//       int startX = floor((trajectories[j][startIndex + 1] - this->minY) / this->resolution) + this->shiftW;
//       int endY = floor((trajectories[j][startIndex + 2] - this->minX) / this->resolution) + this->shiftH;
//       int endX = floor((trajectories[j][startIndex + 3] - this->minY) / this->resolution) + this->shiftW;
//       line(colorProjection,Point(startX,startY),Point(endX,endY),this->trajColors[j],2);
//       startIndex += 2;
//     }
//   }
//   for(int j = 0; j < trajectories.size(); j++) {
//     int markerY = floor((trajectories[j][trajectories[j].size() - 2] - this->minX) / this->resolution) + this->shiftH;
//     int markerX = floor((trajectories[j][trajectories[j].size() - 1] - this->minY) / this->resolution) + this->shiftW;
//     circle(colorProjection,Point(markerX,markerY),12,this->trajColors[j],-1);
//   }
//   return colorProjection.clone();
// }

Mat CloudWallSegmentation::DrawTrajectoriesTime (vector<vector<float>> trajectories, Mat cloudProjection, vector<int> timeSinceStart, int timeLimit){
  while (this->timeStamps.size() < timeSinceStart.size()){
    vector<int> newTimeVec;
    timeStamps.push_back(newTimeVec);
    trajSize.push_back(newTimeVec);
  }
  for(int k = 0; k < timeSinceStart.size(); k++) {
  this->timeStamps[k].push_back(timeSinceStart[k]);
  this->trajSize[k].push_back((trajectories[k].size() - 1) / 2);
  }
  vector<int> startIndex = clearTrajectoryOlderThan(timeLimit);
  Mat colorProjection;
  cvtColor(cloudProjection, colorProjection, CV_GRAY2RGB);
  for(int j = 0; j < trajectories.size(); j++) {
    // for(int i = 0; i < trajectories[j].size() - 4; i += 2) {
    int startind = startIndex[j];
    if (startind % 2 != 0) startind++;
    // int startind = 0;
    while (startind < trajectories[j].size() - 4) {
      int startY = floor((trajectories[j][startind] - this->minX) / this->resolution) + this->shiftH;
      int startX = floor((trajectories[j][startind + 1] - this->minY) / this->resolution) + this->shiftW;
      int endY = floor((trajectories[j][startind + 2] - this->minX) / this->resolution) + this->shiftH;
      int endX = floor((trajectories[j][startind + 3] - this->minY) / this->resolution) + this->shiftW;
      line(colorProjection,Point(startX,startY),Point(endX,endY),this->trajColors[j],2);
      startind += 2;
    }
  }
  for(int j = 0; j < trajectories.size(); j++) {
    int markerY = floor((trajectories[j][trajectories[j].size() - 2] - this->minX) / this->resolution) + this->shiftH;
    int markerX = floor((trajectories[j][trajectories[j].size() - 1] - this->minY) / this->resolution) + this->shiftW;
    circle(colorProjection,Point(markerX,markerY),12,this->trajColors[j],-1);
  }
  return colorProjection.clone();
}

Mat CloudWallSegmentation::DrawTrajectoriesTime2(vector<float> trajectories, vector<int> sizes,Mat cloudProjection, vector<int> timeSinceStart, int timeLimit){
  while (this->timeStamps.size() < timeSinceStart.size()){
    vector<int> newTimeVec;
    timeStamps.push_back(newTimeVec);
    trajSize.push_back(newTimeVec);
  }
  for(int k = 0; k < timeSinceStart.size(); k++) {
    this->timeStamps[k].push_back(timeSinceStart[k]);
    this->trajSize[k].push_back(sizes[k]/2);
  }
  vector<int> startIndex = clearTrajectoryOlderThan(timeLimit);
  Mat colorProjection;
  cvtColor(cloudProjection, colorProjection, CV_GRAY2RGB);
  int offset=0;
  for(int j = 0; j < sizes.size(); j++) {
    // for(int i = 0; i < trajectories[j].size() - 4; i += 2) {
    int startind = startIndex[j];
    if (startind % 2 != 0) startind++;
    //std::cout<<"\nstartind: "<<startind<<"\n";
    while (startind < sizes[j]-2) {
      int startY = floor((trajectories[offset+startind] - this->minX) / this->resolution) + this->shiftH;
      int startX = floor((trajectories[offset+startind + 1] - this->minY) / this->resolution) + this->shiftW;
      int endY = floor((trajectories[offset+startind + 2] - this->minX) / this->resolution) + this->shiftH;
      int endX = floor((trajectories[offset+startind + 3] - this->minY) / this->resolution) + this->shiftW;
      //std::cout<<startX<<", "<<endX<<"; "<<startY<<", "<<endY;
      line(colorProjection,Point(startX,startY),Point(endX,endY),this->trajColors[j],2);
      startind += 2;
    }
    offset+=sizes[j];
  }
  offset=0;
  for(int j = 0; j < sizes.size(); j++) {
    int markerY = floor((trajectories[offset+sizes[j] - 2] - this->minX) / this->resolution) + this->shiftH;
    int markerX = floor((trajectories[offset+sizes[j] - 1] - this->minY) / this->resolution) + this->shiftW;
    circle(colorProjection,Point(markerX,markerY),12,this->trajColors[j],-1);
    offset+=sizes[j];
  }
  return colorProjection.clone();
}

Mat CloudWallSegmentation::DrawTrajectoriesD (vector<vector<double>> trajectories, Mat cloudProjection){
  for(int j = 0; j < trajectories.size(); j++) {
    for(int i = 0; i < trajectories[j].size() - 4; i += 4) {
      int startX = floor((trajectories[j][i] - (double)this->minX) / (double)this->resolution);
      int startY = floor((trajectories[j][i + 1] - (double)this->minY) / (double)this->resolution);
      int endX = floor((trajectories[j][i + 2] - (double)this->minX) / (double)this->resolution);
      int endY = floor((trajectories[j][i + 3] - (double)this->minY) / (double)this->resolution);
      line(cloudProjection,Point(startY,startX),Point(endY,endX),(unsigned char)255,2);
  }
  }
  return cloudProjection.clone();
}

// int CloudWallSegmentation::clearTrajectoryOlderThan(int sec) {
//   int timeIndex = 0;
//   for (int i = this->timeStamps.size() - 1; i >= 0; i--) {
//     int passedTime = timeStamps.back() - timeStamps[i];
//     if (passedTime > sec) {
//       timeIndex = i;
//       break;
//     }
//   }
//   int trajIndex = trajSize[timeIndex];
//   return trajIndex;
// }

vector<int> CloudWallSegmentation::clearTrajectoryOlderThan(int sec) {
  vector<int> timeIndex;
  bool timeAdd = false;
  for(int j = 0; j < this->timeStamps.size(); j++){
    for (int i = this->timeStamps[j].size() - 1; i >= 0; i--) {
      int passedTime = timeStamps[j].back() - timeStamps[j][i];
      if (passedTime > sec) {
        timeIndex.push_back(i);
        timeAdd = true;
        break;
      }
    }
    if(!timeAdd){
      timeIndex.push_back(0);
    }
    else {
      timeAdd = false;
    }
  }
  vector<int> trajIndex;
  for(int i = 0; i < timeIndex.size(); i++) {
    trajIndex.push_back(this->trajSize[i][timeIndex[i]]);
    //cerr << "trajindex " << i << " :" << trajIndex[i] << endl;
  }
  return trajIndex;
}
