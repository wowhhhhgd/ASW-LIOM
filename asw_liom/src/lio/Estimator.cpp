// BSD 3-Clause License

// Copyright (c) 2021, LIVOX
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.

// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.

// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "Estimator/Estimator.h"
#include <random>



int Estimator::s_w_s = 5;
int marg_size = 1;


Estimator::Estimator(const float& filter_corner, const float& filter_surf)
{

    // ========= 更新地图方式 && 回环检测 ============
    numberOfCores = 4;
    surroundingKeyframeSearchRadius = 50.0;
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    surroundingKeyframeDensity = 2.0;
    // for surrounding key poses of scan-to-map optimization
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity,
                                                      surroundingKeyframeDensity, surroundingKeyframeDensity);
    mappingCornerLeafSize = 0.2;
    mappingSurfLeafSize = 0.3;
    downSizeFilterCornerGlobal.setLeafSize(
            mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurfGlobal.setLeafSize(
            mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterICPGlobal.setLeafSize(
            mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

    aLoopIsClosed = false;
    // 开启回环检测
    loopClosureEnableFlag = true;
    historyKeyframeSearchRadius = 25.0;
    historyKeyframeSearchTimeDiff = 30.0;
    historyKeyframeFitnessScore = 0.3;
    historyKeyframeSearchNum = 25;
    loopClosureFrequency = 10;
    


    // ========= 更新地图方式 && 回环检测 ============


    // laserCloudCornerFromLocal.reset(new pcl::PointCloud<PointType>);
    // laserCloudSurfFromLocal.reset(new pcl::PointCloud<PointType>);
    // laserCloudNonFeatureFromLocal.reset(new pcl::PointCloud<PointType>);


    laserCloudCornerLast.resize(SLIDEWINDOWSIZE);
    for(auto& p:laserCloudCornerLast)
        p.reset(new pcl::PointCloud<PointType>);

    laserCloudSurfLast.resize(SLIDEWINDOWSIZE);
    for(auto& p:laserCloudSurfLast)
        p.reset(new pcl::PointCloud<PointType>);

    laserCloudNonFeatureLast.resize(SLIDEWINDOWSIZE);
    for(auto& p:laserCloudNonFeatureLast)
        p.reset(new pcl::PointCloud<PointType>);

    laserCloudCornerStack.resize(SLIDEWINDOWSIZE);
    for(auto& p:laserCloudCornerStack)
        p.reset(new pcl::PointCloud<PointType>);

    laserCloudSurfStack.resize(SLIDEWINDOWSIZE);
    for(auto& p:laserCloudSurfStack)
        p.reset(new pcl::PointCloud<PointType>);

    laserCloudNonFeatureStack.resize(SLIDEWINDOWSIZE);
    for(auto& p:laserCloudNonFeatureStack)
        p.reset(new pcl::PointCloud<PointType>);

    laserCloudCornerForMap.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfForMap.reset(new pcl::PointCloud<PointType>);
    laserCloudNonFeatureForMap.reset(new pcl::PointCloud<PointType>);
    transformForMap.setIdentity();

    kdtreeCornerFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
    kdtreeSurfFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
    kdtreeNonFeatureFromLocal.reset(new pcl::KdTreeFLANN<PointType>);

    for(int i = 0; i < localMapWindowSize; i++){
        localCornerMap[i].reset(new pcl::PointCloud<PointType>);
        localSurfMap[i].reset(new pcl::PointCloud<PointType>);
        localNonFeatureMap[i].reset(new pcl::PointCloud<PointType>);
    }

    downSizeFilterCorner.setLeafSize(filter_corner, filter_corner, filter_corner);
    downSizeFilterSurf.setLeafSize(filter_surf, filter_surf, filter_surf);
    downSizeFilterNonFeature.setLeafSize(0.4, 0.4, 0.4);
    map_manager = new MAP_MANAGER(filter_corner, filter_surf);

    // 更新地图线程
    // threadMap = std::thread(&Estimator::threadMapIncrement, this);   // map update threads
    threadLoop = std::thread(&Estimator::threadLoopClosure, this);

    GlobalConerMapFiltered.reset( new pcl::PointCloud<PointType>);
}

Estimator::~Estimator()
{
    delete map_manager;
}

// 优化位姿
void Estimator::EstimateLidarPose(std::list<LidarFrame>& lidarFrameList,
                           const Eigen::Matrix4d& exTlb,
                           const Eigen::Vector3d& gravity,
                           int lidarMode)
{

    // Lidar和IMU之间的外参
    Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3,3).transpose();
    Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3,1);

    timeLaserInfoCur = lidarFrameList.front().timeStamp;
    
    // 优化过的滑动窗口中第一帧或者最后一帧的位姿
    Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
    // 将位姿从IMU系转换到Lidar系下
    transformTobeMapped.topLeftCorner(3,3) = lidarFrameList.back().Q * exRbl;
    transformTobeMapped.topRightCorner(3,1) = lidarFrameList.back().Q * exPbl 
                                                + lidarFrameList.back().P;

    int stack_count = 0;

    int corner_cnt = 0;
    // push feature point to pointcloud
    // 准备滑动窗口中每个窗口中的特征点
    for(const auto& l : lidarFrameList)
    {
        // normal_z = 1 为角点
        laserCloudCornerLast[stack_count]->clear();
        for(const auto& p : l.laserCloud->points){
            if(std::fabs(p.normal_z - 1.0) < 1e-5){
                laserCloudCornerLast[stack_count]->push_back(p);
                corner_cnt++;
            }
        }

        // normal_z = 2 为面点
        laserCloudSurfLast[stack_count]->clear();
        for(const auto& p : l.laserCloud->points){
            if(std::fabs(p.normal_z - 2.0) < 1e-5)
                laserCloudSurfLast[stack_count]->push_back(p);
        }

        // 这个其实laserCloudNonFeatureLast没有用
        laserCloudNonFeatureLast[stack_count]->clear();
        for(const auto& p : l.laserCloud->points){
            if(std::fabs(p.normal_z - 3.0) < 1e-5)
                laserCloudNonFeatureLast[stack_count]->push_back(p);
        }

        // downsample feature pointcloud
        laserCloudCornerStack[stack_count]->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast[stack_count]);
        downSizeFilterCorner.filter(*laserCloudCornerStack[stack_count]);

        laserCloudSurfStack[stack_count]->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast[stack_count]);
        downSizeFilterSurf.filter(*laserCloudSurfStack[stack_count]);

        laserCloudNonFeatureStack[stack_count]->clear();
        downSizeFilterNonFeature.setInputCloud(laserCloudNonFeatureLast[stack_count]);
        downSizeFilterNonFeature.filter(*laserCloudNonFeatureStack[stack_count]);
        stack_count++;
    }

    
    bool is_degenerate = false;
    bool is_shorter = false;
    // 准备局部地图
    extractSurroundingKeyFrames();
    if(laserCloudCornerFromMapDSNum > 0 && laserCloudSurfFromMapDSNum > 0){
        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
        // 优化滑动窗口中每个窗口对应的位姿
        Estimate(lidarFrameList, exTlb, gravity, is_degenerate, is_shorter);
        std::cout << "local map size, corner: " << laserCloudCornerFromMapDSNum << " surf: "
        << laserCloudSurfFromMapDSNum << std::endl;
    }


    // Make sure system collect engough feature points, Estimate with all pointcloud; or local cloud
    ROS_INFO_STREAM("[Estimator] | laserCloudCornerFromMapDSNum: "  <<
         laserCloudCornerFromMapDSNum << " |laserCloudSurfFromMapDSNum: "<< laserCloudSurfFromMapDSNum);
    
    ROS_INFO_STREAM("[Estimator]  lidarMode: " << lidarMode);
    
    // Save the result; if is not degenerate, use the pose
    // transformTobeMapped = Eigen::Matrix4d::Identity();
    // if(lidarMode == 1 && !is_degenerate && corner_cnt > 200){
    
    // lidarMode = 2
    if(lidarMode == 1  && !is_degenerate && corner_cnt > 100){
        transformTobeMapped.topLeftCorner(3,3) = lidarFrameList.front().Q * exRbl;
        transformTobeMapped.topRightCorner(3,1) = lidarFrameList.front().Q * exPbl 
                                                    + lidarFrameList.front().P;

      
    // }else if(lidarMode == 2 && !is_degenerate && corner_cnt > 50){
    }else if(lidarMode == 2 &&  corner_cnt > 50){

        transformTobeMapped.topLeftCorner(3,3) = lidarFrameList.front().Q * exRbl;
        transformTobeMapped.topRightCorner(3,1) = lidarFrameList.front().Q * exPbl 
                                                    + lidarFrameList.front().P;
    } else{
        if(lidarMode == 1)
            ROS_INFO_STREAM("[Hori] Corner count: "<< corner_cnt << " | In_degenerate Env: " << is_degenerate);
        if(lidarMode == 2)
            ROS_INFO_STREAM("[Velo] Corner count: "<< corner_cnt << " | In_degenerate Env: " << is_degenerate);
        ROS_INFO_STREAM("In lidar degenerate environment, using predicted pose;  corner_cnt: " << corner_cnt);
        Eigen::Vector3d pose;
        pose.x() = lidarFrameList.front().P.x();
        pose.y() = lidarFrameList.front().P.y();
        pose.z() = transformTobeMapped(2,3);
        // std::cout<<  pose.z() << std::endl;
        transformTobeMapped.topRightCorner(3,1) = pose;

        // return;
        // transformTobeMapped.topRightCorner(3,1).x() = lidarFrameList.front().P.x(); // Adopt the x and y from lidar; and z from stereo camera
        // transformTobeMapped.topRightCorner(3,1).y() = lidarFrameList.front().P.y();
    }
    // check the common features
    // 更新局部地图
    if(!is_degenerate) // Not so stable when too few feature
    {
        // Update the map -> Here we go to the global map increment service.
        // 这个地方为什么要上锁呢？因为在更新全局地图的时候会使用到下面的三类特征点
        // std::unique_lock<std::mutex> locker(mtx_Map);
        *laserCloudCornerForMap = *laserCloudCornerStack[0]; // latest scan pointcloud feature
        *laserCloudSurfForMap = *laserCloudSurfStack[0];
        *laserCloudNonFeatureForMap = *laserCloudNonFeatureStack[0];

        // *laserCloudCornerForMap = *laserCloudCornerStack[marg_size - 1]; // latest scan pointcloud feature
        // *laserCloudSurfForMap = *laserCloudSurfStack[marg_size - 1];
        // *laserCloudNonFeatureForMap = *laserCloudNonFeatureStack[marg_size - 1];
        transformForMap = transformTobeMapped;
       
        // lidarMode = 2
        ROS_INFO_STREAM("lidarMode " << lidarMode); 
        if(lidarMode == 1){
            Eigen::Vector3d curret_hori_pose = transformTobeMapped.topRightCorner(3,1);
            Eigen::Vector3d pose_diff = last_velo_update_pose - curret_hori_pose;     // check difference with velo position
            if( pose_diff.dot(pose_diff) >= 0.5){

                ROS_INFO_STREAM("Sliding window size : " << stack_count);
                // MapIncrementLocal(laserCloudCornerForMap,laserCloudSurfForMap,laserCloudNonFeatureForMap,transformTobeMapped);
                last_hori_update_pose = curret_hori_pose;
                ROS_INFO_STREAM("[Hori]Increment map : current pose -> (" << last_hori_update_pose.x() << "," << last_hori_update_pose.y()
                                     << "," << last_hori_update_pose.z() << " | pose diff: " << pose_diff.dot(pose_diff)<<  ")");
            }
        }
        if(lidarMode == 2){
            // laserCloudCornerForMap->clear();
            Eigen::Vector3d curret_velo_pose = transformTobeMapped.topRightCorner(3,1);
            Eigen::Vector3d pose_diff = last_velo_update_pose - curret_velo_pose;
            float dis =  pose_diff.x()*pose_diff.x() + pose_diff.y()*pose_diff.y() + pose_diff.z()*pose_diff.z() ;
            if( dis >= 0.5){
                last_velo_update_pose = curret_velo_pose;
                ROS_INFO_STREAM("[Hori]Increment map : current pose -> (" << last_velo_update_pose.x() << "," << last_velo_update_pose.y()
                                     << "," << last_velo_update_pose.z() << " | pose diff: " << pose_diff.dot(pose_diff)<< " " << dis << ")");
            }

            addOdomFactor();
            addLoopFactor();

            // std::cout << "****************************************************" << std::endl;
            // gtSAMgraph.print("GTSAM Graph:\n");
            // update iSAM
            isam->update(gtSAMgraph, initialEstimate);
            isam->update();

            if (aLoopIsClosed == true){
                isam->update();
                isam->update();
                isam->update();
                isam->update();
                isam->update();
            }

            gtSAMgraph.resize(0);
            initialEstimate.clear();

            // save key poses
            PointType thisPose3D;
            PointTypePose thisPose6D;
            
            Pose3 latestEstimate(transformForMap);

            isamCurrentEstimate = isam->calculateEstimate();
            latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);

            std::cout << "GTSAM esti after " << latestEstimate << std::endl;

            thisPose3D.x = latestEstimate.translation().x();
            thisPose3D.y = latestEstimate.translation().y();
            thisPose3D.z = latestEstimate.translation().z();
            thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
            cloudKeyPoses3D->push_back(thisPose3D);

            thisPose6D.x = thisPose3D.x;
            thisPose6D.y = thisPose3D.y;
            thisPose6D.z = thisPose3D.z;
            thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
            thisPose6D.roll = latestEstimate.rotation().roll();
            thisPose6D.pitch = latestEstimate.rotation().pitch();
            thisPose6D.yaw = latestEstimate.rotation().yaw();
            thisPose6D.time = timeLaserInfoCur;
            cloudKeyPoses6D->push_back(thisPose6D);

            Eigen::Matrix4d trans = latestEstimate.matrix();
            // 从Lidar转到IMU系
            trans.topLeftCorner(3,3) = trans.topLeftCorner(3,3) * exRbl.transpose();
            trans.topRightCorner(3,1) = trans.topRightCorner(3,1) - trans.topLeftCorner(3,3)* exPbl;
            lidarFrameList.front().P.x() = trans(0,3);
            lidarFrameList.front().P.y() = trans(1,3);
            lidarFrameList.front().P.z() = trans(2,3);
            Eigen::Matrix3d rota = trans.topLeftCorner(3,3);
            lidarFrameList.front().Q = rota;

            // save all the received edge and surf points
            pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*laserCloudCornerForMap, *thisCornerKeyFrame);
            pcl::copyPointCloud(*laserCloudSurfForMap, *thisSurfKeyFrame);

            // save key frame cloud
            cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
            surfCloudKeyFrames.push_back(thisSurfKeyFrame);

            correctPoses();

        }
    }


    _fail_detected = is_degenerate;
}

// 拼接局部地图
void Estimator::extractSurroundingKeyFrames(){
    if(cloudKeyPoses3D->points.empty() == true){
        return;
    }

    extractNearby();
}

void Estimator::extractNearby(){
    // 保存xyz三维位置
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis; 

    // 通过kd树查找距离当前位置一定范围内的位置
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
    
    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(),
                                                (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
    // 将这些位置保存下来
    
    for (int i = 0; i < (int)pointSearchInd.size(); ++i){
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }

    // 将位置进行下采样
    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

    // 由于经过下采样，每个位置对应的索引可能会发生变化，因此需要对每个位置的索引进行更新
    for (auto &pt : surroundingKeyPosesDS->points){
        kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
        // intensity存储的是下标索引
        pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
    }

    // also extract some latest key frames in case the robot rotates in one position
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i){
        if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
            surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
        else
            break;
    }

    std::cout << "tadius size:  " << surroundingKeyPosesDS->size() << std::endl;
    extractCloud(surroundingKeyPosesDS);

}

void Estimator::extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract){
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear();

    for(int i = 0; i < (int)cloudToExtract->size(); ++i){
        if(pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back())
                            > surroundingKeyframeSearchRadius){
                                continue;
                            }
        int thisKeyInd = (int)cloudToExtract->points[i].intensity;
        if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()){
            // transformed cloud available
            *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
            *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
        }
        else{
            // transformed cloud not available
            pcl::PointCloud<PointType> laserCloudCornerTemp =
                *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                                         &cloudKeyPoses6D->points[thisKeyInd]);

            pcl::PointCloud<PointType> laserCloudSurfTemp =
                *transformPointCloud(surfCloudKeyFrames[thisKeyInd],
                                         &cloudKeyPoses6D->points[thisKeyInd]);

            *laserCloudCornerFromMap += laserCloudCornerTemp;
            *laserCloudSurfFromMap += laserCloudSurfTemp;
            laserCloudMapContainer[thisKeyInd] = std::make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
        }
    }
    
    // Downsample the surrounding corner key frames (or map)
    downSizeFilterCornerGlobal.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCornerGlobal.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
    // Downsample the surrounding surf key frames (or map)
    downSizeFilterSurfGlobal.setInputCloud(laserCloudSurfFromMap);
    downSizeFilterSurfGlobal.filter(*laserCloudSurfFromMapDS);
    laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

    // clear map cache if too large
    if (laserCloudMapContainer.size() > 1000)
        laserCloudMapContainer.clear();

}

float Estimator::pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

pcl::PointCloud<PointType>::Ptr Estimator::transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                                                        PointTypePose *transformIn){
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur =
        pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z,
                                   transformIn->roll, transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i){

        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x +
                                    transCur(0, 1) * pointFrom.y +
                                    transCur(0, 2) * pointFrom.z +
                                    transCur(0, 3);

        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x +
                                    transCur(1, 1) * pointFrom.y +
                                    transCur(1, 2) * pointFrom.z +
                                    transCur(1, 3);

        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x +
                                    transCur(2, 1) * pointFrom.y +
                                    transCur(2, 2) * pointFrom.z +
                                    transCur(2, 3);

        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;                                                


}


[[noreturn]] void Estimator::threadLoopClosure(){

    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        performLoopClosure();
        visualizeLoopClosure();
    }
}

void Estimator::performLoopClosure(){
    if (cloudKeyPoses3D->points.empty() == true)
        return;

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // find keys
    int loopKeyCur;
    int loopKeyPre;

    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
        return;

    // extract cloud
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    {
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;
        // if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        //     publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        return;

    // publish corrected cloud
    // if (pubIcpKeyFrames.getNumSubscribers() != 0)
    // {
    //     pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
    //     pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
    //     publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    // }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(std::make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    std::cout << "find cloop closure! " << std::endl;
    mtx.unlock();

    // add loop constriant
    loopIndexContainer[loopKeyCur] = loopKeyPre;
}

bool Estimator::detectLoopClosureDistance(int *latestID, int *closestID){
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    // find the closest history key frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i){
        int id = pointSearchIndLoop[i];
        // 时间间隔需要大于一个阈值
        if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff){
            loopKeyPre = id;
            break;
        }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
}

void Estimator::loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes,
                               const int &key, const int &searchNum){
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    for (int i = -searchNum; i <= searchNum; ++i){
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize)
            continue;
        *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear],
                                                   &copy_cloudKeyPoses6D->points[keyNear]);
        *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],
                                                   &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty())
            return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICPGlobal.setInputCloud(nearKeyframes);
    downSizeFilterICPGlobal.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}


void Estimator::visualizeLoopClosure(){

}


// compute the cost function of each line
// 计算点到直线的残差
void Estimator::processPointToLine(std::vector<ceres::CostFunction *>& edges,
                                   std::vector<FeatureLine>& vLineFeatures,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudCornerLocal,
                                   const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d)
{
    ROS_WARN_STREAM("[processPointToLine] Start ... "); 
    Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
    Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
    Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
    if(!vLineFeatures.empty()){
        for(const auto& l : vLineFeatures){
            auto* e = Cost_NavState_IMU_Line::Create(l.pointOri,
                                                    l.lineP1,
                                                    l.lineP2,
                                                    Tbl,
                                                    Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
            edges.push_back(e);
        }
        return;
    }
    ROS_WARN_STREAM("vLineFeatures.empty() "<< vLineFeatures.empty() << " " << laserCloudCorner->points.size());

    PointType _pointOri, _pointSel, _coeff;
    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix< double, 3, 3 > _matA1;
    _matA1.setZero();

    int laserCloudCornerStackNum = laserCloudCorner->points.size();
    pcl::PointCloud<PointType>::Ptr kd_pointcloud(new pcl::PointCloud<PointType>);
    int debug_num1 = 0;
    int debug_num2 = 0;
    int debug_num12 = 0;
    int debug_num22 = 0;
    for (int i = 0; i < laserCloudCornerStackNum; i++)
    {
        _pointOri = laserCloudCorner->points[i];
        MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
        if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) 
            continue;

        if(laserCloudCornerLocal->points.size() > 20 ){
            kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
            if (_pointSearchSqDis2[4] < thres_dist) {

                debug_num2 ++;
                float cx = 0;
                float cy = 0;
                float cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerLocal->points[_pointSearchInd2[j]].x;
                    cy += laserCloudCornerLocal->points[_pointSearchInd2[j]].y;
                    cz += laserCloudCornerLocal->points[_pointSearchInd2[j]].z;
                }
                cx /= 5;
                cy /= 5;
                cz /= 5;

                float a11 = 0;
                float a12 = 0;
                float a13 = 0;
                float a22 = 0;
                float a23 = 0;
                float a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerLocal->points[_pointSearchInd2[j]].x - cx;
                    float ay = laserCloudCornerLocal->points[_pointSearchInd2[j]].y - cy;
                    float az = laserCloudCornerLocal->points[_pointSearchInd2[j]].z - cz;

                    a11 += ax * ax;
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                _matA1(0, 0) = a11;
                _matA1(0, 1) = a12;
                _matA1(0, 2) = a13;
                _matA1(1, 0) = a12;
                _matA1(1, 1) = a22;
                _matA1(1, 2) = a23;
                _matA1(2, 0) = a13;
                _matA1(2, 1) = a23;
                _matA1(2, 2) = a33;

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
                Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

                if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
                    debug_num22++;
                    float x1 = cx + 0.1 * unit_direction[0];
                    float y1 = cy + 0.1 * unit_direction[1];
                    float z1 = cz + 0.1 * unit_direction[2];
                    float x2 = cx - 0.1 * unit_direction[0];
                    float y2 = cy - 0.1 * unit_direction[1];
                    float z2 = cz - 0.1 * unit_direction[2];

                    Eigen::Vector3d tripod1(x1, y1, z1);
                    Eigen::Vector3d tripod2(x2, y2, z2);
                    auto* e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                            tripod1,
                                                            tripod2,
                                                            Tbl,
                                                            Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
                    edges.push_back(e);
                    vLineFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                tripod1,
                                                tripod2);
                    vLineFeatures.back().ComputeError(m4d);
                }
            }
        }

    }
    ROS_WARN_STREAM("[processPointToLine] End ... "); 
}

// 计算点到平面的残差
void Estimator::processPointToPlanVec(std::vector<ceres::CostFunction *>& edges,
                                   std::vector<FeaturePlanVec>& vPlanFeatures,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
                                   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfLocal,
                                   const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                   const Eigen::Matrix4d& exTlb,
                                   const Eigen::Matrix4d& m4d,
                                   bool& is_degenerate)
{
    Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
    Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
    Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
    if(!vPlanFeatures.empty()){
        for(const auto& p : vPlanFeatures){
        auto* e = Cost_NavState_IMU_Plan_Vec::Create(p.pointOri,
                                                    p.pointProj,
                                                    Tbl,
                                                    p.sqrt_info);
        edges.push_back(e);
        }
        return;
    }
    PointType _pointOri, _pointSel, _coeff;
    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix< double, 5, 3 > _matA0;
    _matA0.setZero();
    Eigen::Matrix< double, 5, 1 > _matB0;
    _matB0.setOnes();
    _matB0 *= -1;
    Eigen::Matrix< double, 3, 1 > _matX0;
    _matX0.setZero();
    //获取平面点的数量
    int laserCloudSurfStackNum = laserCloudSurf->points.size();

    int debug_num1 = 0;
    int debug_num2 = 0;
    int debug_num12 = 0;
    int debug_num22 = 0;

    // search  5 nearest pints
    std::vector<Eigen::Vector3d> pNormals;
    for (int i = 0; i < laserCloudSurfStackNum; i++) {
        _pointOri = laserCloudSurf->points[i];
        MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
        if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) 
            continue;

        // 如果全局地图中的点不够用，就用局部地图中的点
        if(laserCloudSurfLocal->points.size() > 20 )
        {
            kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
            if (_pointSearchSqDis2[4] < thres_dist)
            {
                debug_num2++;
                for (int j = 0; j < 5; j++)
                {
                    _matA0(j, 0) = laserCloudSurfLocal->points[_pointSearchInd2[j]].x;
                    _matA0(j, 1) = laserCloudSurfLocal->points[_pointSearchInd2[j]].y;
                    _matA0(j, 2) = laserCloudSurfLocal->points[_pointSearchInd2[j]].z;
                }
                _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

                float pa = _matX0(0, 0);
                float pb = _matX0(1, 0);
                float pc = _matX0(2, 0);
                float pd = 1;

                float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    if (std::fabs(pa * laserCloudSurfLocal->points[_pointSearchInd2[j]].x +
                                pb * laserCloudSurfLocal->points[_pointSearchInd2[j]].y +
                                pc * laserCloudSurfLocal->points[_pointSearchInd2[j]].z + pd) > 0.2) {
                    planeValid = false;
                    break;
                    }
                }

                if (planeValid) {
                    debug_num22 ++;
                    double dist = pa * _pointSel.x +
                                pb * _pointSel.y +
                                pc * _pointSel.z + pd;
                    Eigen::Vector3d omega(pa, pb, pc);
                    pNormals.push_back(omega);
                    Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x,_pointSel.y,_pointSel.z) - (dist * omega);
                    Eigen::Vector3d e1(1, 0, 0);
                    Eigen::Matrix3d J = e1 * omega.transpose();
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
                    Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
                    Eigen::Matrix3d info = (1.0/IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
                    info(1, 1) *= plan_weight_tan;
                    info(2, 2) *= plan_weight_tan;
                    Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

                    auto* e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                                point_proj,
                                                                Tbl,
                                                                sqrt_info);
                    edges.push_back(e);
                    vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                point_proj,
                                                sqrt_info);
                    vPlanFeatures.back().ComputeError(m4d);
                }
            }

        }

    }

    double min_eigen = checkLocalizability(pNormals);
    if(min_eigen < 3.0){
        ROS_WARN_STREAM("In degenerated environment : min_eigen -> " << min_eigen << " < 3.0");
        is_degenerate = true;
    }

}

// 这个函数只有在被边缘化的时候使用
void Estimator::processNonFeatureICP(std::vector<ceres::CostFunction *>& edges,
                                     std::vector<FeatureNon>& vNonFeatures,
                                     const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
                                     const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
                                     const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
                                     const Eigen::Matrix4d& exTlb,
                                     const Eigen::Matrix4d& m4d){
    Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
    Tbl.topLeftCorner(3,3) = exTlb.topLeftCorner(3,3).transpose();
    Tbl.topRightCorner(3,1) = -1.0 * Tbl.topLeftCorner(3,3) * exTlb.topRightCorner(3,1);
    if(!vNonFeatures.empty()){
        for(const auto& p : vNonFeatures){
        auto* e = Cost_NonFeature_ICP::Create(p.pointOri,
                                                p.pa,
                                                p.pb,
                                                p.pc,
                                                p.pd,
                                                Tbl,
                                                Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
        edges.push_back(e);
        }
        return;
    }

    PointType _pointOri, _pointSel, _coeff;
    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix< double, 5, 3 > _matA0;
    _matA0.setZero();
    Eigen::Matrix< double, 5, 1 > _matB0;
    _matB0.setOnes();
    _matB0 *= -1;
    Eigen::Matrix< double, 3, 1 > _matX0;
    _matX0.setZero();

    int laserCloudNonFeatureStackNum = laserCloudNonFeature->points.size();
    for (int i = 0; i < laserCloudNonFeatureStackNum; i++) {
        _pointOri = laserCloudNonFeature->points[i];
        MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
        int id = map_manager->FindUsedNonFeatureMap(&_pointSel,laserCenWidth_last,laserCenHeight_last,laserCenDepth_last);

        if(id == 5000) continue;

        if(std::isnan(_pointSel.x) || std::isnan(_pointSel.y) ||std::isnan(_pointSel.z)) continue;

        if(GlobalNonFeatureMap[id].points.size() > 100) {

            NonFeatureKdMap[id].nearestKSearch(_pointSel, 5, _pointSearchInd, _pointSearchSqDis);

            if (_pointSearchSqDis[4] < 1 * thres_dist) {
                for (int j = 0; j < 5; j++) {
                    _matA0(j, 0) = GlobalNonFeatureMap[id].points[_pointSearchInd[j]].x;
                    _matA0(j, 1) = GlobalNonFeatureMap[id].points[_pointSearchInd[j]].y;
                    _matA0(j, 2) = GlobalNonFeatureMap[id].points[_pointSearchInd[j]].z;
                }
                _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

                float pa = _matX0(0, 0);
                float pb = _matX0(1, 0);
                float pc = _matX0(2, 0);
                float pd = 1;

                float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (std::fabs(pa * GlobalNonFeatureMap[id].points[_pointSearchInd[j]].x +
                                    pb * GlobalNonFeatureMap[id].points[_pointSearchInd[j]].y +
                                    pc * GlobalNonFeatureMap[id].points[_pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if(planeValid) {

                    auto* e = Cost_NonFeature_ICP::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                            pa,
                                                            pb,
                                                            pc,
                                                            pd,
                                                            Tbl,
                                                            Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
                    edges.push_back(e);
                    vNonFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd);
                    vNonFeatures.back().ComputeError(m4d);

                    continue;
                }
            }

        }

        if(laserCloudNonFeatureLocal->points.size() > 20 ){
            kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
            if (_pointSearchSqDis2[4] < 1 * thres_dist) {
                for (int j = 0; j < 5; j++) {
                    _matA0(j, 0) = laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].x;
                    _matA0(j, 1) = laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].y;
                    _matA0(j, 2) = laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].z;
                }
                _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

                float pa = _matX0(0, 0);
                float pb = _matX0(1, 0);
                float pc = _matX0(2, 0);
                float pd = 1;

                float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (std::fabs(pa * laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].x +
                                    pb * laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].y +
                                    pc * laserCloudNonFeatureLocal->points[_pointSearchInd2[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if(planeValid) {

                    auto* e = Cost_NonFeature_ICP::Create(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                            pa,
                                                            pb,
                                                            pc,
                                                            pd,
                                                            Tbl,
                                                            Eigen::Matrix<double, 1, 1>(1/IMUIntegrator::lidar_m));
                    edges.push_back(e);
                    vNonFeatures.emplace_back(Eigen::Vector3d(_pointOri.x,_pointOri.y,_pointOri.z),
                                                pa,
                                                pb,
                                                pc,
                                                pd);
                    vNonFeatures.back().ComputeError(m4d);
                }
            }
        }
    }

}

double Estimator::checkLocalizability( std::vector<Eigen::Vector3d> planeNormals){
    // ROS_INFO_STREAM("[Estimator::LocalizabilityCheck]");
    // Transform it into Eigen::matrixXd
    Eigen::MatrixXd mat;
    if( planeNormals.size() > 10){
        mat.setZero(planeNormals.size(), 3);
        for(int i = 0; i < planeNormals.size(); i++)
        {
            mat(i,0) = planeNormals[i].x();
            mat(i,1) = planeNormals[i].y();
            mat(i,2) = planeNormals[i].z();
            // std::cout<<"mat " << i << ": " <<  mat(i,0) << " ," << mat(i,1) << " " << mat(i,2) << std::endl;
        }

        // SVD, get constraint strength
        Eigen::JacobiSVD<Eigen::MatrixXd > svd(planeNormals.size(), 3);
        svd.compute(mat);
        if(svd.singularValues().z() < 2.0){
            _fail_detected = true;
            ROS_WARN_STREAM("Low convincing result-> singular values: " << svd.singularValues().x() << " " << svd.singularValues().y() << " " << svd.singularValues().z());
        }
        return svd.singularValues().z();
    }else{
        _fail_detected = true;
        ROS_WARN_STREAM(" Too few normal vector received -> " << planeNormals.size());
        return -1;
    }

    _fail_detected = false;
}

bool Estimator::failureDetected(){
    return _fail_detected;
}


void Estimator::vector2double(const std::list<LidarFrame>& lidarFrameList){
    int i = 0;
    for(const auto& l : lidarFrameList){
        Eigen::Map<Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
        PR.segment<3>(0) = l.P;
        PR.segment<3>(3) = Sophus::SO3d(l.Q).log();

        Eigen::Map<Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
        VBias.segment<3>(0) = l.V;
        VBias.segment<3>(3) = l.bg;
        VBias.segment<3>(6) = l.ba;
        i++;
    }
}

// 将优化参数从数组转换为四元数
void Estimator::double2vector(std::list<LidarFrame>& lidarFrameList){
    int i = 0;
    for(auto& l : lidarFrameList){
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
        Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
        l.P = PR.segment<3>(0);
        l.Q = Sophus::SO3d::exp(PR.segment<3>(3)).unit_quaternion();
        l.V = VBias.segment<3>(0);
        l.bg = VBias.segment<3>(3);
        l.ba = VBias.segment<3>(6);
        i++;
    }
}


// 将变换矩阵转换为gtsam格式
gtsam::Pose3 Estimator::trans2gtsamPose(Eigen::Matrix4d m){
    return gtsam::Pose3(m);
}

Eigen::Affine3f Estimator::pclPointToAffine3f(PointTypePose thisPoint){
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z,
                                      thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

gtsam::Pose3 Estimator::pclPointTogtsamPose3(PointTypePose thisPoint){

    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
    
}

void Estimator::addOdomFactor(){
    if (cloudKeyPoses3D->points.empty())
        {
            // rad*rad, meter*meter
            noiseModel::Diagonal::shared_ptr priorNoise =
                noiseModel::Diagonal::Variances(
                    (Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());

            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformForMap), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformForMap));
            std::cout << "cloudKeyPoses3D->points.empty(): " << cloudKeyPoses3D->points.empty() << std::endl;
        }
        else
        {
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances(
                (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo = trans2gtsamPose(transformForMap);
            std::cout << "poseFrom: " << poseFrom << std::endl;
            std::cout << "poseTo: " << poseTo << std::endl;
            
            gtSAMgraph.add(BetweenFactor<Pose3>(
                cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
            std::cout << "cloudKeyPoses3D->points.size(): " << cloudKeyPoses3D->points.size() << std::endl;;
        }
}

void Estimator::addLoopFactor(){
    if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            std::cout << "add loop factor: " << indexFrom << "-" << indexTo << "\n"
            << poseBetween << std::endl;
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
}

void Estimator::correctPoses(){
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true){
        // clear map cache
        laserCloudMapContainer.clear();
        // clear path
        globalPath.poses.clear();
        // update key poses
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i){

            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll =
                isamCurrentEstimate.at<Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch =
                isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw =
                isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

            updatePath(cloudKeyPoses6D->points[i]);
        }

        aLoopIsClosed = false;
    }

}

void Estimator::updatePath(const PointTypePose &pose_in){
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id = "lio_world";
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}


double Estimator::GetWindowSize(double x){
    return x*x + x + 1;
}


void Estimator::Estimate(std::list<LidarFrame>& lidarFrameList,
                         const Eigen::Matrix4d& exTlb,
                         const Eigen::Vector3d& gravity,
                         bool& is_degenerate,
                         bool& is_shorter)
{
    ROS_INFO_STREAM("Estimator::Estimate windowSize lidarFrameList.size() : " << lidarFrameList.size());

    static uint32_t frame_count = 0;

    // 滑动窗口的大小
    int windowSize = lidarFrameList.size();
    Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
    
    // Lidar和IMU之间的外参
    Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3,3).transpose();
    Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3,1);


    // store point to line features 点到线的结构体数组
    std::vector<std::vector<FeatureLine>> vLineFeatures(windowSize);
    for(auto& v : vLineFeatures){
        v.reserve(2000);
    }

    // store point to plan features 点到面的结构体数组
    std::vector<std::vector<FeaturePlanVec>> vPlanFeatures(windowSize);
    for(auto& v : vPlanFeatures){
        v.reserve(2000);
    }

    std::vector<std::vector<FeaturePlan>> ToPlanFeatures(windowSize);
    for(auto& v : vPlanFeatures){
        v.reserve(2000);
    }

    std::vector<std::vector<FeatureNon>> vNonFeatures(windowSize);
    for(auto& v : vNonFeatures){
        v.reserve(2000);
    }

    // if(windowSize == SLIDEWINDOWSIZE) {
    if(windowSize == Estimator::get_s_w_s()) {
        plan_weight_tan = 0.0003;
        thres_dist = 1.0;
    } else {
        plan_weight_tan = 0.0;
        thres_dist = 25.0;
    }
    // excute optimize process
    const int max_iters = 5;
    for(int iterOpt = 0; iterOpt < max_iters; ++iterOpt){
        
        // 将优化参数初始化
        vector2double(lidarFrameList);

        // lossfunction 其实就是对costfunction的残差的二次函数，这里使用huberloss
        ceres::LossFunction* loss_function = NULL;
        loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);

        // if(windowSize == SLIDEWINDOWSIZE) {
        if(windowSize == Estimator::get_s_w_s()) {
            loss_function = NULL;
        } else {
            loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
        }

        // 定义ceres problem
        ceres::Problem::Options problem_options;
        ceres::Problem problem(problem_options);
        // 设置求解的一些设置
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = 10;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 6;
        ceres::Solver::Summary summary;

        std::cout << "before add para" << std::endl;
        // for(int i = 0; i < windowSize; ++i) {
        //     problem.AddParameterBlock(para_PR[i], 6);   // add pose, orientation
        // }

        // for(int i = 0; i < windowSize; ++i){
        //     problem.AddParameterBlock(para_VBias[i], 9);    // add bias
        // }
        // 添加参数块
        for(int i = 0; i < windowSize; ++i) {
            // x、y、z、roll、pitch、yaw
            problem.AddParameterBlock(para_PR[i], 6);   // add pose, orientation
            // vx、vy、vz、bx、by、bz
            problem.AddParameterBlock(para_VBias[i], 9);    // add velocity bias
        }
        std::cout << "end add para" << std::endl;

        /* ================= 调试代码 =================
        std::cout << "before problem. the para: " <<std::endl;
        for(int i = 0; i < windowSize; ++i) {
            std::cout << para_PR[i][0] << " " 
                      << para_PR[i][1] << " " 
                      << para_PR[i][2] << " " 
                      << para_PR[i][3] << " " 
                      << para_PR[i][4] << " " 
                      << para_PR[i][5] << std::endl;
        }
        for(int i = 0; i < windowSize; ++i){
            std::cout << para_VBias[i][0] << " " 
                      << para_VBias[i][1] << " " 
                      << para_VBias[i][2] << " " 
                      << para_VBias[i][3] << " " 
                      << para_VBias[i][4] << " " 
                      << para_VBias[i][5] << " "
                      << para_VBias[i][6] << " " 
                      << para_VBias[i][7] << " " 
                      << para_VBias[i][8] << std::endl;
        }
         ================= 调试代码 =================    */
        
        // 记录下滑动窗口中最后一个窗口的位姿
        Eigen::Quaterniond q_before_opti = lidarFrameList.back().Q;
        Eigen::Vector3d t_before_opti = lidarFrameList.back().P;

        // 定义存储损失函数的容器
        std::vector<std::vector<ceres::CostFunction *>> edgesLine(windowSize);
        std::vector<std::vector<ceres::CostFunction *>> edgesPlan(windowSize);
        std::vector<std::vector<ceres::CostFunction *>> edgesNon(windowSize);
        std::thread threads[3];

        // ================= multi threads to compute cost function of each point =================
        // 计算点到线的距离和点到面的距离
        for(int f = 0; f < windowSize; ++f) {
            auto frame_curr = lidarFrameList.begin();
            // 将 it 迭代器前进或后退 n 个位置
            std::advance(frame_curr, f);
            transformTobeMapped = Eigen::Matrix4d::Identity();
            // 初始化
            transformTobeMapped.topLeftCorner(3,3) = frame_curr->Q * exRbl;
            transformTobeMapped.topRightCorner(3,1) = frame_curr->Q * exPbl + frame_curr->P;
            threads[0] = std::thread(&Estimator::processPointToLine, this,
                                    std::ref(edgesLine[f]),
                                    std::ref(vLineFeatures[f]),
                                    std::ref(laserCloudCornerStack[f]),
                                    std::ref(laserCloudCornerFromMapDS),
                                    std::ref(kdtreeCornerFromMap),
                                    std::ref(exTlb),
                                    std::ref(transformTobeMapped));
            threads[1] = std::thread(&Estimator::processPointToPlanVec, this,
                                    std::ref(edgesPlan[f]),
                                    std::ref(vPlanFeatures[f]),
                                    std::ref(laserCloudSurfStack[f]),
                                    std::ref(laserCloudSurfFromMapDS),
                                    std::ref(kdtreeSurfFromMap),
                                    std::ref(exTlb),
                                    std::ref(transformTobeMapped),
                                    std::ref(is_degenerate));
            threads[0].join();
            threads[1].join();
        }
        // ================= multi threads to compute cost function of each point =================
        
        int cntSurf = 0;
        int cntCorner = 0;
        int cntNon = 0;


        std::vector<ceres::ResidualBlockId> residual_id_vec;
        // 存储优化变量的数组
        std::vector<double *> para_vec;
        para_vec.push_back(para_PR[windowSize - 1]);


        // for(int f = 0; f < windowSize; ++f){
        //     para_vec.push_back(para_PR[f]);
        // }

        // add cost function of feature points
        // =================== 添加点线和点面残差 ===================
        std::cout << "before add feature cost function,residual_id_vec.size(): "<< residual_id_vec.size() << std::endl;
        // if(windowSize == SLIDEWINDOWSIZE) {
        // 现在窗口已经满了，需要滑动了
        if(windowSize == Estimator::get_s_w_s()) {
            // Add constraints to solver
            thres_dist = 1.0;

            // 第一次循环先标记需要将哪些残差添加进problem中
            if(iterOpt == 0)
            {
                for(int f = 0; f < windowSize; ++f){
                    
                    int cntFtu = 0;
                    for (auto &e : edgesLine[f]) {
                        if(std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5){
                            ceres::ResidualBlockId id = problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            if(f == (windowSize -1)){
                                // std::cout << "residual_id_vec.push_back(id);" << std::endl;
                                residual_id_vec.push_back(id);
                            }
                            // else{
                            //     std::cout << "f: " << f << std::endl;
                            // }

                            // residual_id_vec.push_back(id);
                            
                            vLineFeatures[f][cntFtu].valid = true;
                        }else{
                            vLineFeatures[f][cntFtu].valid = false;
                        }
                        cntFtu++;
                        cntCorner++;
                    }

                    cntFtu = 0;
                    for (auto &e : edgesPlan[f]) {
                        if(std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5){

                            // problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            ceres::ResidualBlockId id = problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            // if(f == windowSize -1){
                            //     residual_id_vec.push_back(id);
                            // }

                            vPlanFeatures[f][cntFtu].valid = true;
                        }else{
                            vPlanFeatures[f][cntFtu].valid = false;
                        }
                        cntFtu++;
                        cntSurf++;
                    }

                   
                }
            }else{
                for(int f = 0; f < windowSize; ++f){
                    int cntFtu = 0;
                    for (auto &e : edgesLine[f]) {
                        if(vLineFeatures[f][cntFtu].valid) {
                            ceres::ResidualBlockId id = problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            if(f == (windowSize -1)){
                                //  std::cout << "residual_id_vec.push_back(id);" << std::endl;
                                residual_id_vec.push_back(id);
                            }
                            // residual_id_vec.push_back(id);
                        }
                        cntFtu++;
                        cntCorner++;
                    } 
                    cntFtu = 0;
                    for (auto &e : edgesPlan[f]) {
                        if(vPlanFeatures[f][cntFtu].valid){
                            // problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            ceres::ResidualBlockId id = problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            // if(f == windowSize -1){
                            //     residual_id_vec.push_back(id);
                            // }
                        }
                        cntFtu++;
                        cntSurf++;
                    }

                    // cntFtu = 0;
                    // for (auto &e : edgesNon[f]) {
                    //     if(vNonFeatures[f][cntFtu].valid){
                    //         problem.AddResidualBlock(e, loss_function, para_PR[f]);
                    //     }
                    //     cntFtu++;
                    //     cntNon++;
                    // }
                }
            }
        } else {
            if(iterOpt == 0) {
                thres_dist = 10.0;
            } else {
                thres_dist = 1.0;
            }

            for(int f = 0; f < windowSize; ++f){
                int cntFtu = 0;
                for (auto &e : edgesLine[f]) {
                    if(std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5){
                        ceres::ResidualBlockId id = problem.AddResidualBlock(e, loss_function, para_PR[f]);
                        // residual_id_vec.push_back(id);
                        vLineFeatures[f][cntFtu].valid = true;
                    }else{
                        vLineFeatures[f][cntFtu].valid = false;
                    }
                    cntFtu++;
                    cntCorner++;
                }
                cntFtu = 0;
                for (auto &e : edgesPlan[f]) {
                    if(std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5){
                        ceres::ResidualBlockId id = problem.AddResidualBlock(e, loss_function, para_PR[f]);
                        
                        vPlanFeatures[f][cntFtu].valid = true;
                    }else{
                        vPlanFeatures[f][cntFtu].valid = false;
                    }
                    cntFtu++;
                    cntSurf++;
                }

                // cntFtu = 0;
                // for (auto &e : edgesNon[f]) {
                //     if(std::fabs(vNonFeatures[f][cntFtu].error) > 1e-5){
                //         problem.AddResidualBlock(e, loss_function, para_PR[f]);
                //         vNonFeatures[f][cntFtu].valid = true;
                //     }else{
                //         vNonFeatures[f][cntFtu].valid = false;
                //     }
                //     cntFtu++;
                //     cntNon++;
                // }
            }
        }
        std::cout << "after add feature cost function,residual_id_vec.size(): "
        << residual_id_vec.size() <<  std::endl;
        // =================== 添加点线和点面残差 ===================

        // if(is_degenerate){
        //     ROS_WARN_STREAM("Degenerated environment, still update map");
        //     // return;
        // }

        // add IMU CostFunction
        // =============== 添加IMU残差 ===============
        std::cout << "before add imu costfunction" << std::endl;
        for(int f = 1; f < windowSize; ++f){
            auto frame_curr = lidarFrameList.begin();
            std::advance(frame_curr, f);
            problem.AddResidualBlock(Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,
                                                                    const_cast<Eigen::Vector3d&>(gravity),
                                                                    Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                                                                            (frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                            .matrixL().transpose()),
                                    nullptr,
                                    para_PR[f-1],
                                    para_VBias[f-1],
                                    para_PR[f],
                                    para_VBias[f]);
        }
        std::cout << "after add imu costfunction" << std::endl;
        // =============== 添加IMU残差 ===============

        // 记录有可能被修改的窗口大小
        static int old_window_size = Estimator::get_s_w_s();

        // ===============  添加边缘化残差 ===============
        std::cout << "before add new marg factor" << std::endl;
        // if ((old_window_size >= Estimator::get_s_w_s()) && last_marginalization_info){
        if ((windowSize == Estimator::get_s_w_s()) && last_marginalization_info){
            // construct new marginlization_factor
            auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            std::cout << "windowSize == Estimator::get_s_w_s()" << std::endl;
            std::cout << "last_marginalization_parameter_blocks.size: " 
            << last_marginalization_parameter_blocks.size() << std::endl;
            problem.AddResidualBlock(marginalization_factor, nullptr,
                                    last_marginalization_parameter_blocks);
        }else{
            std::cout << "old_window_size < Estimator::get_s_w_s() "
            << old_window_size << " < " << Estimator::get_s_w_s()
            << " and windowSize < Estimator::get_s_w_s()" 
            <<  windowSize << " < "  << Estimator::get_s_w_s() << std::endl;
        }
        std::cout << "after add new marg factor" << std::endl;
        // ===============  添加边缘化残差 ===============

        // 进行求解
        ceres::Solve(options, &problem, &summary);

        /* ================= 调试代码 =================  
        // std::cout << "after problem. the para: " <<std::endl;
        // for(int i = 0; i < windowSize; ++i) {
        //     std::cout << para_PR[i][0] << " " 
        //               << para_PR[i][1] << " " 
        //               << para_PR[i][2] << " " 
        //               << para_PR[i][3] << " " 
        //               << para_PR[i][4] << " " 
        //               << para_PR[i][5] << std::endl;
        // }
        // for(int i = 0; i < windowSize; ++i){
        //     std::cout << para_VBias[i][0] << " " 
        //               << para_VBias[i][1] << " " 
        //               << para_VBias[i][2] << " " 
        //               << para_VBias[i][3] << " " 
        //               << para_VBias[i][4] << " " 
        //               << para_VBias[i][5] << " "
        //               << para_VBias[i][6] << " " 
        //               << para_VBias[i][7] << " " 
        //               << para_VBias[i][8] << std::endl;
        // }
        // ================= 调试代码 =================  
        */
        
        std::cout   << " Summary initial_cost: " << summary.initial_cost
                    << "| final_cost: " << summary.final_cost
                    << "| IsSolutionUsable(): " << summary.IsSolutionUsable()
                    << std::endl;

        double2vector(lidarFrameList);

        Eigen::Quaterniond q_after_opti = lidarFrameList.back().Q;
        Eigen::Vector3d t_after_opti = lidarFrameList.back().P;
        Eigen::Vector3d V_after_opti = lidarFrameList.back().V;
        double deltaR = (q_before_opti.angularDistance(q_after_opti)) * 180.0 / M_PI;
        double deltaT = (t_before_opti - t_after_opti).norm();

        // ====================== 计算位置的协方差矩阵 以及求解雅克比 ==================
        // 如果优化完成，计算滑动窗口中最后一个位姿的xyz的协方差的迹,作为调整窗口的依据
        if(deltaR < 0.05 && deltaT < 0.05 || (iterOpt+1) == max_iters){

            // 获取雅克比矩阵
            evaluateBA(problem, summary, residual_id_vec, para_vec);

            // Eigen::Matrix<double,3,3, Eigen::RowMajor> cov_pose = Eigen::Matrix<double,3,3, Eigen::RowMajor>::Zero();
            // ceres::Covariance::Options cov_options;
            // ceres::Covariance covariance(cov_options);
            // std::vector<std::pair<const double*, const double*>> covariance_blocks;
            // int frame_size = lidarFrameList.size();
            // covariance_blocks.push_back(std::make_pair(para_PR[frame_size - 1], para_PR[frame_size - 1]));
            // // for(int i = 0; i < lidarFrameList.size(); i++){
            // //     covariance_blocks.push_back(std::make_pair(para_PR[i], para_PR[i]));
            // // }
            // cov_pose.setZero();
            // Eigen::Matrix<double,6,6, Eigen::RowMajor> Ceres_cov_mid = Eigen::Matrix<double,6,6, Eigen::RowMajor>::Zero();
            // if( covariance.Compute(covariance_blocks, &problem) )
            // {
            //     covariance.GetCovarianceBlockInTangentSpace(para_PR[frame_size - 1],para_PR[frame_size - 1], Ceres_cov_mid.data()); 
            //     cov_pose = Ceres_cov_mid.block<3, 3>(0, 0);
            // }
            // // static int cout_s_w_w = 0;
            // // cout_s_w_w++;
            // std::cout << "cov_pose.diagonal() = \n" << cov_pose.diagonal() << std::endl;
            // double trace_cov = cov_pose.trace();
            // cov_pose_vec.push_back(trace_cov);
            // std::cout << "cov_pose.trace:  " << trace_cov << std::endl;
        }
        // ====================== 计算位置的协方差矩阵 以及求解雅克比 ================== 
        
        

        // Add marginalization parameter blocks
        // 获取边缘化之后的因子，作为下次优化的先验
        if (deltaR < 0.05 && deltaT < 0.05 || (iterOpt+1) == max_iters){
            // ROS_INFO("Frame: %d\n",frame_count++);
            // if(windowSize != SLIDEWINDOWSIZE) break; // break here
            if(windowSize != Estimator::get_s_w_s()){
                std::cout << windowSize << " != " << Estimator::get_s_w_s() << " break"<< std::endl;
                break; // break here
            } 
                

            marg_size = 1;
            
            // 使用随机设备作为种子
            std::random_device rd;
            // 使用 Mersenne Twister 引擎
            std::mt19937 gen(rd());
            // 定义分布范围为5到15之间的整数
            std::uniform_int_distribution<int> dis(5, 15);
            
            // static std::chrono::steady_clock::time_point lastChangeTime = std::chrono::steady_clock::now();
            static auto lastChangeTime = std::chrono::high_resolution_clock::now();
            // 获取当前时间
            auto currentTime = std::chrono::high_resolution_clock::now();

            // 如果距离上次改变randomSize的时间超过了一段时间间隔（60秒），则改变randomSize的值
            double dt = 1.e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime-lastChangeTime).count();
            double currentTimeSec = currentTime.time_since_epoch().count() * 1.e-9;
            double lastChangeTimeSec = lastChangeTime.time_since_epoch().count() * 1.e-9;
            std::cout << "currentTime: " << currentTimeSec << std::endl;
            std::cout << "lastChangeTimeSec: " << lastChangeTimeSec << std::endl;

            std::cout << "dt: " << dt << " s" << std::endl;
            std::cout << "jaco_pose_vec.size(): " << jaco_pose_vec.size() << std::endl;
            // 间隔delt时间进行一次窗长的调整
            
            if (dt >= 60) {
                // 更新上次改变a的时间
                lastChangeTime = currentTime;
                
                int randomSize = dis(gen);
                // int randomSize = 0;
                old_window_size = Estimator::get_s_w_s();

                double x1 = jaco_pose_vec[jaco_pose_vec.size() - 2];
                double x2 = jaco_pose_vec.back();
                if(x2 - x1 > 0){
                    randomSize  = old_window_size + 1;
                    if(randomSize > 6){
                        randomSize = old_window_size;
                    }
                }else{
                    randomSize = old_window_size - 1;
                    if(randomSize < 3){
                        randomSize = old_window_size;
                    }
                }

                if(randomSize > old_window_size){
                    std::cout << "randomSize >= Estimator::get_s_w_s() "  
                    << randomSize << " > " << Estimator::get_s_w_s() << std::endl;
                    Estimator::set_s_w_s(randomSize);

                    std::cout << "randomSize的值已经从 "<< old_window_size << " 改变为 " << randomSize 
                    << " 滑动窗长变长，不需要边缘化, break, 重新来过"<< std::endl;
                    delete last_marginalization_info;
                    last_marginalization_info = nullptr;
                    for(int f = 0; f < windowSize; ++f){
                        edgesLine[f].clear();
                        edgesPlan[f].clear();
                        edgesNon[f].clear();
                        vLineFeatures[f].clear();
                        vPlanFeatures[f].clear();
                        vNonFeatures[f].clear();
                    }
                    
                    break;
                }else if(randomSize < old_window_size){
                    int delt_size = Estimator::get_s_w_s() - randomSize;
                    std::cout << "randomSize < Estimator::get_s_w_s() "  
                    << randomSize << " < " << Estimator::get_s_w_s() 
                    << " delt_size " << delt_size << std::endl;
                    marg_size = delt_size + 1;
                    delete last_marginalization_info;
                    last_marginalization_info = nullptr;
                    Estimator::set_s_w_s(randomSize);
                    std::cout << "randomSize的值已经从 "<< old_window_size << " 改变为 "  << randomSize 
                    << " 滑动窗长变短，需要边缘化掉 "<<  marg_size << " 帧, break, 重新来过" << std::endl;
                    for(int f = 0; f < windowSize; ++f){
                        edgesLine[f].clear();
                        edgesPlan[f].clear();
                        edgesNon[f].clear();
                        vLineFeatures[f].clear();
                        vPlanFeatures[f].clear();
                        vNonFeatures[f].clear();
                    }
                    break; // 重新来过
                }else{
                    marg_size = 1;
                    std::cout << "randomSize的值没有改变,继续优化" << randomSize << std::endl;
                }
                
                
            }else{
                marg_size = 1;
                std::cout << "间隔时间不到, 不改变窗长,继续优化 " << std::endl;

            }
            
            // apply marginalization
            std::cout << "apply marginalization, marg_size = " << marg_size << std::endl;
            auto *marginalization_info = new MarginalizationInfo();

            

            // 第一次运行到这里，不会执行
            if (last_marginalization_info){
                
                std::cout << "before add last_marginalization_info factor" << std::endl;
                std::vector<int> drop_set;

                for (int marg_idx = 0; marg_idx < marg_size; marg_idx++){
                    for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++){
                        if (last_marginalization_parameter_blocks[i] == para_PR[marg_idx] ||
                            last_marginalization_parameter_blocks[i] == para_VBias[marg_idx])
                            drop_set.push_back(i);
                    }
                }

                std::cout << "drop_set.size: " << drop_set.size() << std::endl;
                auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                auto *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                                last_marginalization_parameter_blocks,
                                                                drop_set);
                marginalization_info->addResidualBlockInfo(residual_block_info);
                std::cout << "after add last_marginalization_info factor" << std::endl;
            }

            std::cout << "before add feature residualinfo and IMU residualinfo" << std::endl;
            
            for(int advance_step = 1; advance_step <= marg_size; advance_step++){
                auto frame_curr = lidarFrameList.begin();
                std::advance(frame_curr, advance_step);
                std::cout << "before Cost_NavState_PRV_Bias::Create" << std::endl;
                ceres::CostFunction* IMU_Cost = Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,
                                                                                const_cast<Eigen::Vector3d&>(gravity),
                                                                                Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                                                                                        (frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                                        .matrixL().transpose());
                std::cout << "end Cost_NavState_PRV_Bias::Create" << std::endl;
                // for(int p_i = 0; p_i = IMU_Cost->parameter_block_sizes().size(); p_i++)                                                                        
                //     std::cout << "IMU_Cost parameter_block_sizes[ "<< p_i <<
                //     " ] = " << IMU_Cost->parameter_block_sizes()[p_i]
                //     << std::endl;

                auto *residual_block_info = new ResidualBlockInfo(IMU_Cost, nullptr,
                                                                    std::vector<double *>{para_PR[advance_step-1], para_VBias[advance_step-1], para_PR[advance_step], para_VBias[advance_step]},
                                                                    std::vector<int>{0, 1}); // 这里就是要边缘化的参数id
                marginalization_info->addResidualBlockInfo(residual_block_info);
                std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();
                for(int i = 0; i < parameter_block_sizes.size(); i++){
                    std::cout <<"i: "<< parameter_block_sizes[i] << std::endl;
                }

                int f = advance_step - 1;
                transformTobeMapped = Eigen::Matrix4d::Identity();
                transformTobeMapped.topLeftCorner(3,3) = frame_curr->Q * exRbl;
                transformTobeMapped.topRightCorner(3,1) = frame_curr->Q * exPbl + frame_curr->P;
                edgesLine[f].clear();
                edgesPlan[f].clear();
                // edgesNon[f].clear();
                threads[0] = std::thread(&Estimator::processPointToLine, this,
                                        std::ref(edgesLine[f]),
                                        std::ref(vLineFeatures[f]),
                                        std::ref(laserCloudCornerStack[f]),
                                        std::ref(laserCloudCornerFromMapDS),
                                        std::ref(kdtreeCornerFromMap),
                                        std::ref(exTlb),
                                        std::ref(transformTobeMapped));

                threads[1] = std::thread(&Estimator::processPointToPlanVec, this,
                                        std::ref(edgesPlan[f]),
                                        std::ref(vPlanFeatures[f]),
                                        std::ref(laserCloudSurfStack[f]),
                                        std::ref(laserCloudSurfFromMapDS),
                                        std::ref(kdtreeSurfFromMap),
                                        std::ref(exTlb),
                                        std::ref(transformTobeMapped),
                                        std::ref(is_degenerate));

                threads[0].join();
                threads[1].join();

                int cntFtu = 0;
                for (auto &e : edgesLine[f]) {
                    if(vLineFeatures[f][cntFtu].valid){
                        auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                                            std::vector<double *>{para_PR[f]},
                                                                            std::vector<int>{0});
                        
                        // for(int p_i = 0; p_i = e->parameter_block_sizes().size(); p_i++)                                                                        
                        //     std::cout << "edgesLine parameter_block_sizes[ "<< p_i <<
                        //     " ] = " << e->parameter_block_sizes()[p_i]
                        //     << std::endl;
                        

                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    cntFtu++;
                }

                cntFtu = 0;
                for (auto &e : edgesPlan[f]) {
                    if(vPlanFeatures[f][cntFtu].valid){
                        auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                                            std::vector<double *>{para_PR[f]},
                                                                            std::vector<int>{0});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    cntFtu++;
                }

                std::cout << "apply marg: " << advance_step << " time" << std::endl; 
            }
            std::cout << "after add feature residualinfo and IMU residualinfo" << std::endl;


            std::cout << "before preMarginalize" << std::endl;
            marginalization_info->preMarginalize();
            std::cout << "after preMarginalize" << std::endl;

            std::cout << "before marginalize" << std::endl;
            marginalization_info->marginalize();
            std::cout << "after marginalize" << std::endl;

            std::unordered_map<long, double *> addr_shift;
            // for (int i = marg_size; i < SLIDEWINDOWSIZE; i++)
            for (int i = marg_size; i < old_window_size; i++)
            {
                addr_shift[reinterpret_cast<long>(para_PR[i])] = para_PR[i - 1];
                addr_shift[reinterpret_cast<long>(para_VBias[i])] = para_VBias[i - 1];
            }

            std::cout << "before get parameterblocks" << std::endl;
            std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            std::cout << "after get parameterblocks" << std::endl;

            delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            // 有几个残差块儿，一个窗口有两个残差块儿 para_PR 和 para_VBias
            // 其中para_PR有6个数字，para_VBias有9个数字，一个窗口有15个数字
            last_marginalization_parameter_blocks = parameter_blocks;
            // marg_size = 2, size = 2
            // marg_size = 1, size = 
            std::cout << "parameter_blocks size: " << parameter_blocks.size() << std::endl;
            break;
        }

        if(windowSize !=  Estimator::get_s_w_s()) {
            std::cout << "windowSize " << windowSize << " != " << "old_window_size " << old_window_size 
            << " vLineFeatures clear"<< std::endl;
            for(int f = 0; f < windowSize; ++f){
                edgesLine[f].clear();
                edgesPlan[f].clear();
                edgesNon[f].clear();
                vLineFeatures[f].clear();
                vPlanFeatures[f].clear();
                vNonFeatures[f].clear();
            }
        }
    }

}

Eigen::MatrixXd Estimator::CRSMatrix2EigenMatrix(ceres::CRSMatrix *jacobian_crs_matrix)
{
    Eigen::MatrixXd J(jacobian_crs_matrix->num_rows, jacobian_crs_matrix->num_cols);
    J.setZero();

    std::vector<int> jacobian_crs_matrix_rows, jacobian_crs_matrix_cols;
    std::vector<double> jacobian_crs_matrix_values;
    jacobian_crs_matrix_rows = jacobian_crs_matrix->rows;
    jacobian_crs_matrix_cols = jacobian_crs_matrix->cols;
    jacobian_crs_matrix_values = jacobian_crs_matrix->values;

    int cur_index_in_cols_and_values = 0;
    // rows is a num_rows + 1 sized array
    int row_size = static_cast<int>(jacobian_crs_matrix_rows.size()) - 1;
    // outer loop traverse rows, inner loop traverse cols and values
    for (int row_index = 0; row_index < row_size; ++row_index)
    {
        while (cur_index_in_cols_and_values < jacobian_crs_matrix_rows[row_index + 1])
        {
            J(row_index, jacobian_crs_matrix_cols[cur_index_in_cols_and_values]) = jacobian_crs_matrix_values[cur_index_in_cols_and_values];
            cur_index_in_cols_and_values++;
        }
    }
    return J;
}

void Estimator::evaluateBA(ceres::Problem& problem, const ceres::Solver::Summary& summary, 
        std::vector<ceres::ResidualBlockId>& res_id_vec, std::vector<double *>& para_vec){
    // std::cout << summary.FullReport() << std::endl;

    ceres::Problem::EvaluateOptions EvalOpts;
    EvalOpts.residual_blocks = res_id_vec;
    EvalOpts.parameter_blocks = para_vec;
    
    ceres::CRSMatrix jacobian_crs_matrix;
    std::vector<double> resi;
    problem.Evaluate(EvalOpts, nullptr, &resi, nullptr, &jacobian_crs_matrix);

    // TicToc t_convert_J;
    Eigen::MatrixXd J = CRSMatrix2EigenMatrix(&jacobian_crs_matrix);
    Eigen::MatrixXd J_pos = J.leftCols(3);
    std::cout << "J_pos size: " << J_pos.rows() << "*" << J_pos.cols() << std::endl;
    Eigen::Matrix3d J_pos_trans_J_pos = J_pos.transpose()*J_pos;
    std::cout << "J_pos.transpose()*J_pos \n" << J_pos_trans_J_pos << std::endl;
    J_pos_trans_J_pos = J_pos_trans_J_pos.inverse();
    std::cout << "J_pos.transpose()*J_pos.inverse() \n" << J_pos_trans_J_pos.inverse() << std::endl;
    double trace_J_pos = J_pos_trans_J_pos.trace();
     std::cout << "J_pos.transpose()*J_pos.inverse().trace \n" << trace_J_pos << std::endl;
    double root_sign_trace = std::sqrt(trace_J_pos);
    jaco_pose_vec.push_back(root_sign_trace);

    static double MAX = INT16_MIN;
    static double MIN = INT16_MAX;

    std::cout << "std::sqrt(trace_J_pos)\n" << root_sign_trace << std::endl;
    if(root_sign_trace > MAX){
        MAX = root_sign_trace;
    }

    if(root_sign_trace < MIN){
        MIN = root_sign_trace;
    }

    std::cout << "MAX: " << MAX 
    << " MIN: " << MIN << std::endl;

    std::cout << "residual.size: " << resi.size() << std::endl; 
    for(int i = 0; i < 10; i++){
        std::cout << "residual: " << resi[i] << std::endl;
    }
    std::cout << "problem.NumResidualBlocks(): " << problem.NumResidualBlocks() << std::endl;
    std::cout << "problem.NumResiduals(): " << problem.NumResiduals() << std::endl;
    std::cout << "evaluateBA Jacobian.size: " << J.rows() << " x " << J.cols() << std::endl; 
    // std::cout << "convert sparse matrix cost " << t_convert_J.toc() << std::endl;

    // TicToc t_construct_H;
    Eigen::MatrixXd H = J.transpose()*J;
    // std::cout << "construct H cost " << t_construct_H.toc() << std::endl;

    static std::ofstream J_file("/home/mm_loam/jacobian.txt");
    J_file << std::fixed;
    J_file.precision(9);

    int J_rows = 10, J_cols = J.cols();
    J_file << J_rows << "x" << J_cols << std::endl;
    for(int i = 0; i < J_rows; ++i){
        for(int j = 0; j < J_cols; ++j){
            // if(j >= J_cols-14 && j <= J_cols-12 )
            //     J_file << J(i, j);
            J_file << J(i, j);
            if((j+1) % 6 == 0){
                J_file << std::endl;
            }
            if(j == J_cols - 1){
                J_file << std::endl;
                J_file << std::endl;
            }
            else{
                J_file << " ";
            }
        }
    }

    static std::ofstream H_file("/home/mm_loam/hessian.txt");
    H_file << std::fixed;
    H_file.precision(9);

    int H_rows = H.rows(), H_cols = H.cols();
    for(int i = 0; i < H_rows; ++i){
        for(int j = 0; j < H_cols; ++j){
            H_file << H(i, j);
            if(j == H_cols - 1){
                H_file << std::endl;
                H_file << std::endl;
            }
            else{
                H_file << " ";
            }
        }
    }
}

