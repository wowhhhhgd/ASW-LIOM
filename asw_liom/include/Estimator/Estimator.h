#ifndef LIO_LIVOX_ESTIMATOR_H
#define LIO_LIVOX_ESTIMATOR_H

#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <std_msgs/Float64.h>
#include <std_msgs/Int8.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <queue>
#include <iterator>
#include <future>
#include "MapManager/Map_Manager.h"
#include "utils/ceresfunc.h"
#include "IMUIntegrator/IMUIntegrator.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <chrono>
#include <fstream>
#include <iostream>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;
typedef pcl::PointXYZINormal PointType;

class Estimator{
	
public:
	/** \brief slide window size */
	static const int SLIDEWINDOWSIZE = 20;
	

	/** \brief lidar frame struct */
	struct LidarFrame
	{
		pcl::PointCloud<PointType>::Ptr laserCloud;
		IMUIntegrator imuIntegrator;
		Eigen::Vector3d P; 	   // pose
		Eigen::Vector3d V;	   // speed
		Eigen::Quaterniond Q;  // orientation
		Eigen::Vector3d bg;	   //
		Eigen::Vector3d ba;
		double timeStamp;
		double timeOffset;
		int    lidarType;     // 1 Hori; 2 Velo

		LidarFrame(){
			P.setZero();
			V.setZero();
			Q.setIdentity();
			bg.setZero();
			ba.setZero();
			timeStamp = 0;
			timeOffset = 0;
			lidarType = 0;
		}
	};

	/** \brief point to line feature */
	struct FeatureLine
	{
		Eigen::Vector3d pointOri;
		Eigen::Vector3d lineP1;
		Eigen::Vector3d lineP2;
		double error;
		bool valid;
		FeatureLine(Eigen::Vector3d  po, Eigen::Vector3d  p1, Eigen::Vector3d  p2)
						:pointOri(std::move(po)), lineP1(std::move(p1)), lineP2(std::move(p2)){
			valid = false;
			error = 0;
		}
		double ComputeError(const Eigen::Matrix4d& pose){
			Eigen::Vector3d P_to_Map = pose.topLeftCorner(3,3) * pointOri + pose.topRightCorner(3,1);
			double l12 = std::sqrt((lineP1(0) - lineP2(0))*(lineP1(0) - lineP2(0)) + (lineP1(1) - lineP2(1))*
																						(lineP1(1) - lineP2(1)) + (lineP1(2) - lineP2(2))*(lineP1(2) - lineP2(2)));
			double a012 = std::sqrt(
							((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1)))
							* ((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1)))
							+ ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2)))
								* ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2)))
							+ ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2)))
								* ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))));
			error = a012 / l12;
		}
	};

	/** \brief point to plan feature */
	struct FeaturePlan{
		Eigen::Vector3d pointOri;
		double pa;
		double pb;
		double pc;
		double pd;
		double error;
		bool valid;
		FeaturePlan(const Eigen::Vector3d& po, const double& pa_, const double& pb_, const double& pc_, const double& pd_)
						:pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_){
			valid = false;
			error = 0;
		}
		double ComputeError(const Eigen::Matrix4d& pose){
			Eigen::Vector3d P_to_Map = pose.topLeftCorner(3,3) * pointOri + pose.topRightCorner(3,1);
			error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
		}
	};

	/** \brief point to plan feature */
	struct FeaturePlanVec{
		Eigen::Vector3d pointOri;
		Eigen::Vector3d pointProj;
		Eigen::Matrix3d sqrt_info;
		double error;
		bool valid;
		FeaturePlanVec(const Eigen::Vector3d& po, const Eigen::Vector3d& p_proj, Eigen::Matrix3d sqrt_info_)
						:pointOri(po), pointProj(p_proj), sqrt_info(sqrt_info_) {
			valid = false;
			error = 0;
		}
		double ComputeError(const Eigen::Matrix4d& pose){
			Eigen::Vector3d P_to_Map = pose.topLeftCorner(3,3) * pointOri + pose.topRightCorner(3,1);
			error = (P_to_Map - pointProj).norm();
		}
	};

	/** \brief non feature */
	struct FeatureNon{
		Eigen::Vector3d pointOri;
		double pa;
		double pb;
		double pc;
		double pd;
		double error;
		bool valid;
		FeatureNon(const Eigen::Vector3d& po, const double& pa_, const double& pb_, const double& pc_, const double& pd_)
						:pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_){
			valid = false;
			error = 0;
		}
		double ComputeError(const Eigen::Matrix4d& pose){
			Eigen::Vector3d P_to_Map = pose.topLeftCorner(3,3) * pointOri + pose.topRightCorner(3,1);
			error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
		}
	};

public:
	/** \brief constructor of Estimator
	*/
	Estimator(const float& filter_corner, const float& filter_surf);

	~Estimator();

		/** \brief Open a independent thread to increment MAP cloud
		*/
	
	[[noreturn]] void threadLoopClosure();

    	// 提取关键帧
	void extractSurroundingKeyFrames();
	// 寻找距离当前帧50米和10秒以内的pose，以及pose所代表的corner，surface点云
	void extractNearby();
	// 根据索引拼接点云
	void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract);
	float pointDistance(PointType p1, PointType p2);
	pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                                                        PointTypePose *transformIn);
	void addOdomFactor();
	void addLoopFactor();
	gtsam::Pose3 trans2gtsamPose(Eigen::Matrix4d m);
	gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint);
	void performLoopClosure();
	void visualizeLoopClosure();
	void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes,
                               const int &key, const int &searchNum);
	bool detectLoopClosureDistance(int *latestID, int *closestID);
	Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint);

	void updatePath(const PointTypePose &pose_in);
	void correctPoses();


	static void set_s_w_s(int size){
		s_w_s = size;
	}
	static int get_s_w_s(){
		return s_w_s;
	}

	/** \brief construct sharp feature Ceres Costfunctions
	* \param[in] edges: store costfunctions
	* \param[in] m4d: lidar pose, represented by matrix 4X4
	*/
	void processPointToLine(std::vector<ceres::CostFunction *>& edges,
							std::vector<FeatureLine>& vLineFeatures,
							const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
							const pcl::PointCloud<PointType>::Ptr& laserCloudCornerMap,
							const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
							const Eigen::Matrix4d& exTlb,
							const Eigen::Matrix4d& m4d);

	/** \brief construct Plan feature Ceres Costfunctions
	* \param[in] edges: store costfunctions
	* \param[in] m4d: lidar pose, represented by matrix 4X4
	*/
	void processPointToPlan(std::vector<ceres::CostFunction *>& edges,
							std::vector<FeaturePlan>& vPlanFeatures,
							const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
							const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
							const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
							const Eigen::Matrix4d& exTlb,
							const Eigen::Matrix4d& m4d);

	void processPointToPlanVec(std::vector<ceres::CostFunction *>& edges,
							   std::vector<FeaturePlanVec>& vPlanFeatures,
							   const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
							   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
							   const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
							   const Eigen::Matrix4d& exTlb,
							   const Eigen::Matrix4d& m4d,
							   bool& is_degenerate);

	void processNonFeatureICP(std::vector<ceres::CostFunction *>& edges,
							  std::vector<FeatureNon>& vNonFeatures,
							  const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
							  const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
							  const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
							  const Eigen::Matrix4d& exTlb,
							  const Eigen::Matrix4d& m4d);

	/** \brief Transform Lidar Pose in slidewindow to double array
		* \param[in] lidarFrameList: Lidar Poses in slidewindow
		*/
	void vector2double(const std::list<LidarFrame>& lidarFrameList);

	/** \brief Transform double array to Lidar Pose in slidewindow
		* \param[in] lidarFrameList: Lidar Poses in slidewindow
		*/
	void double2vector(std::list<LidarFrame>& lidarFrameList);

	/** \brief estimate lidar pose by matchistd::ng current lidar cloud with map cloud and tightly coupled IMU message
		* \param[in] lidarFrameList: multi-frames of lidar cloud and lidar pose
		* \param[in] exTlb: extrinsic matrix between lidar and IMU
		* \param[in] gravity: gravity vector
		*/
	void EstimateLidarPose(std::list<LidarFrame>& lidarFrameList,
						   const Eigen::Matrix4d& exTlb,
						   const Eigen::Vector3d& gravity,
						   int lidarMode);

	void Estimate(std::list<LidarFrame>& lidarFrameList,
				  const Eigen::Matrix4d& exTlb,
				  const Eigen::Vector3d& gravity,
				  bool& is_degenerate,
				  bool& is_shorter);

	pcl::PointCloud<PointType>::Ptr get_corner_map(){
		return map_manager->get_corner_map();
	}
	pcl::PointCloud<PointType>::Ptr get_surf_map(){
		return map_manager->get_surf_map();
	}
	pcl::PointCloud<PointType>::Ptr get_nonfeature_map(){
		return map_manager->get_nonfeature_map();
	}

	


	void MapCornerFeatureFilter(pcl::PointCloud<PointType>::Ptr& map_corner_feature,
								pcl::PointCloud<PointType>::Ptr& map_corner_feature_filtered,
								pcl::PointCloud<PointType>::Ptr& curr_hori_corner,
								pcl::PointCloud<PointType>::Ptr& curr_hori_plane)
	{
		pcl::KdTreeFLANN<PointType>  hori_corner_kdtree;
		pcl::KdTreeFLANN<PointType>  hori_plane_kdtree;
		map_corner_feature_filtered.reset(new pcl::PointCloud<PointType>);

		hori_corner_kdtree.setInputCloud(map_corner_feature);
		hori_plane_kdtree.setInputCloud(curr_hori_plane);

		int points_num = map_corner_feature->points.size();
		PointType point_sel;
		std::vector<int>   cornerPointSearchInd;
    		std::vector<float> cornerPointSearchSqDis;
		std::vector<int>   planePointSearchInd;
    		std::vector<float> planePointSearchSqDis;

		for( int i = 0; i < points_num; i ++){
			point_sel.x = map_corner_feature->points[i].x;
			point_sel.y = map_corner_feature->points[i].y;
			point_sel.z = map_corner_feature->points[i].z;

			hori_corner_kdtree.nearestKSearch(point_sel, 1, cornerPointSearchInd, cornerPointSearchSqDis);
			hori_plane_kdtree.nearestKSearch(point_sel, 4, planePointSearchInd, planePointSearchSqDis);

			if(planePointSearchSqDis[4] < 0.1 && cornerPointSearchSqDis[0] < 0.1){
				map_corner_feature_filtered->push_back(map_corner_feature->points[i]);
			}
		}
	}


	double checkLocalizability( std::vector<Eigen::Vector3d> planeNormals);

	pcl::PointCloud<PointType>::Ptr get_filtered_corner_map(){
		return GlobalConerMapFiltered ;
	}

	bool   failureDetected();
	double 	GetWindowSize(double x);

	// ceres CRSMatrix to EigenMatrix
	Eigen::MatrixXd CRSMatrix2EigenMatrix(ceres::CRSMatrix *jacobian_crs_matrix);
	void evaluateBA(ceres::Problem& problem, const ceres::Solver::Summary& summary, 
				std::vector<ceres::ResidualBlockId>& res_id_vec, std::vector<double *>& para_vec);

	// =============== 不再使用的函数 =====================
	// [[noreturn]] void threadMapIncrement();
	// void MapIncrementLocal(const pcl::PointCloud<PointType>::Ptr& laserCloudCornerStack,
	// 					   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfStack,
	// 					   const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureStack,
	// 					   const Eigen::Matrix4d& transformTobeMapped
	// 					    );


	
private:
	// customized window size 
	static int s_w_s;

	/** \brief store map points */
	MAP_MANAGER* map_manager;

	double para_PR[SLIDEWINDOWSIZE][6];
	double para_VBias[SLIDEWINDOWSIZE][9];
	// double **para_PR;
	// double **para_VBias;

	MarginalizationInfo *last_marginalization_info = nullptr;
	std::vector<double *> last_marginalization_parameter_blocks;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerLast;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfLast;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudNonFeatureLast;

	// corner feature points within local window (50 frames)
	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromLocal;  
	// surf feature points within local window (50 frames)
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromLocal;
	// non feature points within local window (50 frames)	
	pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureFromLocal; 
	
	// latest corner feature scans in sliding windows
	pcl::PointCloud<PointType>::Ptr laserCloudCornerForMap;
	// latest surf   feature scans in sliding windows     
	pcl::PointCloud<PointType>::Ptr laserCloudSurfForMap;	
	// latest non-   feature scans in sliding windows	
	pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureForMap; 

	
	Eigen::Matrix4d transformForMap;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerStack; // vector of framelist of feature pointcloud
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfStack;   // vector of framelist of feature pointcloud
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudNonFeatureStack; // vector of framelist of feature pointcloud
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromLocal;         // kdtree of all feature points  within local window (50 frames)
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromLocal;           // kdtree of all feature points  within local window (50 frames)
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeNonFeatureFromLocal;     // kdtree of all feature points  within local window (50 frames)

	pcl::VoxelGrid<PointType> downSizeFilterCorner;  	// global voxelgrid for corner
	pcl::VoxelGrid<PointType> downSizeFilterSurf; 		// global voxelgrid for surf
	pcl::VoxelGrid<PointType> downSizeFilterNonFeature; // global voxelgrid for NonFeature
	std::mutex mtx_Map;
	std::thread threadMap;
	
	// 特征点的KDtree
	pcl::KdTreeFLANN<PointType> CornerKdMap[10000];
	pcl::KdTreeFLANN<PointType> SurfKdMap[10000];
	pcl::KdTreeFLANN<PointType> NonFeatureKdMap[10000];

	pcl::PointCloud<PointType> GlobalSurfMap[10000];
	pcl::PointCloud<PointType> GlobalCornerMap[10000];
	pcl::PointCloud<PointType> GlobalNonFeatureMap[10000];

	pcl::PointCloud<PointType>::Ptr GlobalConerMapFiltered;

	int laserCenWidth_last = 10;
	int laserCenHeight_last = 5;
	int laserCenDepth_last = 10;
	
	// 局部地图保留帧数
	static const int localMapWindowSize = 50;
	int localMapID = 0;
	// 局部角点地图
	pcl::PointCloud<PointType>::Ptr localCornerMap[localMapWindowSize];
	// 局部面点地图
	pcl::PointCloud<PointType>::Ptr localSurfMap[localMapWindowSize];
	// 局部非特征点地图
	pcl::PointCloud<PointType>::Ptr localNonFeatureMap[localMapWindowSize];

	int map_update_ID = 0;

	int map_skip_frame = 2; //every map_skip_frame frame update map
	double plan_weight_tan = 0.0;
	double thres_dist = 1.0;

	bool _fail_detected = false;
	Eigen::Vector3d last_velo_update_pose = {-1.0, -1.0, -1.0};
	Eigen::Vector3d last_hori_update_pose = {-1.0, -1.0, -1.0};

	std::vector<double> cov_pose_vec;
	std::vector<double> jaco_pose_vec;


	// ========================== 回环检测和地图管理方式 ================
	std::thread threadLoop;
	// CPU Params
    	int numberOfCores;
	double surroundingKeyframeSearchRadius;
	float surroundingKeyframeDensity;
	pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
	pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
	std::mutex mtx;
	pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
	pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    	pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

	// for surrounding key poses of scan-to-map optimization
    	pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;

	double timeLaserInfoCur;

	// 由角点和面点组成的局部地图
	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

	pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

	int laserCloudCornerFromMapDSNum;
    	int laserCloudSurfFromMapDSNum;

	// 时间戳 对应的角点点云和面点点云
    	std::map<int, std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;

	float mappingCornerLeafSize;
    	float mappingSurfLeafSize;
	pcl::VoxelGrid<PointType> downSizeFilterCornerGlobal;
	pcl::VoxelGrid<PointType> downSizeFilterSurfGlobal;
	pcl::VoxelGrid<PointType> downSizeFilterICPGlobal;

	// 存放每一个关键帧
	std::vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    	std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

	ISAM2 *isam;
	NonlinearFactorGraph gtSAMgraph;
	Values initialEstimate;
	Values isamCurrentEstimate;

	bool aLoopIsClosed;
	bool loopClosureEnableFlag;
	// from new to old
	std::map<int, int> loopIndexContainer;
	std::vector<std::pair<int, int>> loopIndexQueue;
	std::vector<gtsam::Pose3> loopPoseQueue;
	std::vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

	float historyKeyframeSearchRadius;
	float historyKeyframeSearchTimeDiff;
	float historyKeyframeFitnessScore;
	int historyKeyframeSearchNum;
	float loopClosureFrequency;

	nav_msgs::Path globalPath;
};

#endif //LIO_LIVOX_ESTIMATOR_H
