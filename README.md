# ASW_LIOM
In this paper, we propose a multi-modal LiDAR inertial odometry and mapping framework that integrates mechanical and solid-state LiDAR to enhance odometry accuracy and mapping density. 
By fusing complementary feature points, our approach leverages the strengths of distinct scanning patterns.
To further improve real-time performance, an adaptive sliding window strategy is presented to adjust window length based on the current LiDAR feature distribution. 
Additionally, we incorporate loop closure detection in the mapping process to minimize cumulative drift.
Extensive experiments conducted on both public and self-collected datasets demonstrate the effectiveness of our method.
# Dependency
PCL version: 1.8.1 

Eigen3 version: 3.3.4 

OpenCV version: 3.2.0

Ceres version: 2.1.0

GTSAM version: 4.0.2

# Install
```
catkin_make -DCATKIN_WHITELIST_PACKAGES="livox_ros_driver"
```

```
catkin_make -DCATKIN_WHITELIST_PACKAGES="union_cloud"
```

```
catkin_make -DCATKIN_WHITELIST_PACKAGES="asw_liom"
```
# Run with public dataset
```
. devel/setup.bash
```
```
roslaunch asw_liom run_asw_liom.launch 
```
Please refer to [tiers-lidars-dataset](https://github.com/TIERS/tiers-lidars-dataset) and [tiers-lidars-dataset-enhanced](tiers-lidars-dataset-enhanced) to play rosbag.

# Thanks
Thanks the open source code :

[multi-modal-loam](https://github.com/TIERS/multi-modal-loam)

[LIO-SAM](https://github.com/TixiaoShan/LIO-SAM)
