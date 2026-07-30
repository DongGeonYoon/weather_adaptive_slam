// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com
// Architecture: FAST-LIO2 baseline with weather-adaptive LiDAR confidence,
// explicit frame-level residual covariance, and confidence-aware iKD-Tree updates.

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <cmath>
#include <thread>
#include <fstream>
#include <csignal>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <cctype>
#include <atomic>
#include <unordered_map>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#ifdef FAST_LIO_USE_CUDA
#include "rank_cuda.h"
#endif

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool   map_pub_en = true;
bool   traj_save_en = false;
// Robustness module switches:
// - rank_weight_enable computes LiDAR point quality ranks.
// - rank_map_filter_enable protects the iKD-Tree map by dropping low-rank points.
// - rank_huber_enable downweights large point-to-plane residuals.
// - rank_frame_confidence_enable applies a frame-level confidence weight.
bool   rank_weight_enable = true, rank_huber_enable = false, rank_frame_confidence_enable = false, rank_map_filter_enable = true;
bool   rank_terminal_distribution_enable = false;
bool   rank_gpu_enable = true, rank_gpu_active = false;
bool   adaptive_lidar_cov_enable = false;
bool   map_confidence_fail_safe_enable = true;
bool   experiment_log_enable = true;
bool   init_gyro_bias_from_mean = true;
bool   lidar_qos_reliable = false, imu_qos_reliable = false;
bool   range_image_pub_en = true, range_image_cloud_pub_en = true, filtered_cloud_pub_en = true;
int    lidar_qos_depth = 50, imu_qos_depth = 5000, executor_threads = 4;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;
double rank_horizontal_resolution_deg = 0.2;
double rank_sigma_r = 1.0;
string rank_range_factor_mode = "linear_boost";
int rank_range_factor_mode_id = 0;
double rank_min_weight = 0.2;
double rank_max_weight = 2.0;
double rank_huber_delta = 0.2;
double rank_map_filter_percent = 20.0;
double rank_fov_up_deg = 2.0;
double rank_fov_down_deg = -24.8;
double rank_frame_alpha_rank = 0.5;
double rank_frame_alpha_count = 0.3;
double rank_frame_alpha_residual = 0.2;
double rank_frame_gamma = 2.0;
double rank_frame_ref_points = 1000.0;
double rank_frame_residual_sigma = 0.1;
double rank_frame_ref_mean = 1.5;
double rank_terminal_distribution_range_bin_size = 10.0;
double rank_terminal_distribution_azimuth_bin_deg = 15.0;
double rank_terminal_distribution_max_range = 0.0;
double map_publish_period_sec = 3.0;
double map_visual_voxel_size = 0.75;
double range_image_max_range = 0.0;
double range_image_cloud_radius = 15.0;
double range_image_cloud_height = 5.0;
double range_image_cloud_z_offset = 0.0;
int map_visual_max_points = 250000;
int range_image_publish_every_n_frames = 1;
int range_image_row_scale = 8;
double frame_rank_weight = 1.0;
double lidar_confidence_last = 1.0;
double lidar_degradation_last = 0.0;
double residual_variance_last = 0.0;
double base_lidar_cov = LASER_POINT_COV;
double adaptive_lidar_cov = LASER_POINT_COV;
double adaptive_cov_min_confidence = 0.1;
double adaptive_cov_degradation_gamma = 4.0;
double adaptive_cov_min = LASER_POINT_COV;
double adaptive_cov_max = 0.1;
string adaptive_cov_mode = "degradation";
int map_min_effective_points = 100;
int map_skip_below_effective_points = 20;
double map_min_rank_valid_ratio = 0.2;
double map_threshold_relax_factor = 0.5;
double rank_mean_last = 0.0;
float rank_map_threshold_last = 0.0f;
int rank_valid_num_last = 0;
int rank_log_every_n_frames = 20;
int rank_terminal_distribution_every_n_frames = 20;
double rank_ring_valid_warning_ratio = 0.9;
double rank_valid_ratio_last = 0.0;
double rank_compute_time_last = 0.0;
double preprocess_time_last = 0.0;
double lidar_gap_last = 0.0;
double last_processed_lidar_time = -1.0;
std::size_t lidar_queue_size_last = 0;
std::size_t imu_queue_size_last = 0;
int imu_samples_last = 0;
bool current_ring_field_present = false;
double current_ring_valid_ratio = 0.0;
string current_ring_field_name;
std::atomic<std::uint64_t> lidar_received_count{0};
std::atomic<std::uint64_t> lidar_processed_count{0};
std::atomic<std::uint64_t> imu_received_count{0};

template <typename T>
T clamp_value(const T &value, const T &low, const T &high)
{
    return std::max(low, std::min(value, high));
}

mutex mtx_buffer;
condition_variable sig_buffer;

string resolve_root_dir()
{
    const char *workspace = std::getenv("FAST_LIO_WS");
    if (workspace != nullptr && workspace[0] != '\0')
    {
        return string(workspace) + "/src/FAST_LIO/";
    }
    return ROOT_DIR;
}

string root_dir = resolve_root_dir();
string map_file_path, lid_topic, imu_topic;
string traj_file_path;
string dataset_name = "generic";
string lidar_frame_id, imu_frame_id;
string map_frame_id = "camera_init", body_frame_id = "body";
string rank_ring_source = "normal_x";
string experiment_log_path;
string range_image_topic = "/debug/cylindrical_range_image";
string range_image_cloud_topic = "/debug/cylindrical_range_cloud";
string filtered_cloud_topic = "/cloud_weather_filtered";

enum RankRingSourceMode
{
    RANK_RING_NORMAL_X = 0,
    RANK_RING_POINTCLOUD2 = 1,
    RANK_RING_ELEVATION = 2
};
int rank_ring_source_mode = RANK_RING_NORMAL_X;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, publish_count = 0;
std::atomic<int> scan_count{0};
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    is_first_lidar = true;

vector<vector<int>>  pointSearchInd_surf;
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
vector<float>        point_ranks;
vector<float>        point_weights;
vector<bool>         point_rank_valid;
vector<float>        effective_point_weights;
vector<float>        effective_point_ranks;
vector<float>        latest_raw_point_ranks;
vector<bool>         latest_raw_point_rank_valid;
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
deque<bool>                       lidar_ring_present_buffer;
deque<double>                     lidar_ring_ratio_buffer;
deque<string>                     lidar_ring_name_buffer;
deque<double>                     lidar_preprocess_time_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                         // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                              // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                              // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
    const int scan_id = ++scan_count;
    ++lidar_received_count;
    if (!lidar_frame_id.empty() && msg->header.frame_id != lidar_frame_id && scan_id % 100 == 0)
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                    "[%s] LiDAR frame mismatch: expected=%s received=%s",
                    dataset_name.c_str(), lidar_frame_id.c_str(), msg->header.frame_id.c_str());

    const double cur_time = get_time_sec(msg->header.stamp);
    const double preprocess_start_time = omp_get_wtime();
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    const double preprocess_time = omp_get_wtime() - preprocess_start_time;
    const bool ring_present = p_pre->last_ring_field_present;
    const double ring_ratio = p_pre->last_ring_valid_ratio;
    const string ring_name = p_pre->last_ring_field_name;

    std::lock_guard<mutex> lock(mtx_buffer);
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
        time_buffer.clear();
        lidar_ring_present_buffer.clear();
        lidar_ring_ratio_buffer.clear();
        lidar_ring_name_buffer.clear();
        lidar_preprocess_time_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    lidar_ring_present_buffer.push_back(ring_present);
    lidar_ring_ratio_buffer.push_back(ring_ratio);
    lidar_ring_name_buffer.push_back(ring_name);
    lidar_preprocess_time_buffer.push_back(preprocess_time);
    last_timestamp_lidar = cur_time;
    if (scan_id < MAXN)
        s_plot11[scan_id] = preprocess_time;
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg)
{
    const int scan_id = ++scan_count;
    ++lidar_received_count;
    const double cur_time = get_time_sec(msg->header.stamp);
    const double preprocess_start_time = omp_get_wtime();
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    const double preprocess_time = omp_get_wtime() - preprocess_start_time;

    std::lock_guard<mutex> lock(mtx_buffer);
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
        time_buffer.clear();
        lidar_ring_present_buffer.clear();
        lidar_ring_ratio_buffer.clear();
        lidar_ring_name_buffer.clear();
        lidar_preprocess_time_buffer.clear();
    }
    if(is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    lidar_ring_present_buffer.push_back(false);
    lidar_ring_ratio_buffer.push_back(0.0);
    lidar_ring_name_buffer.emplace_back();
    lidar_preprocess_time_buffer.push_back(preprocess_time);

    if (scan_id < MAXN)
        s_plot11[scan_id] = preprocess_time;
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    if (!imu_frame_id.empty() && msg_in->header.frame_id != imu_frame_id && publish_count % 2000 == 0)
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                    "[%s] IMU frame mismatch: expected=%s received=%s",
                    dataset_name.c_str(), imu_frame_id.c_str(), msg_in->header.frame_id.c_str());
    publish_count ++;
    ++imu_received_count;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    std::lock_guard<mutex> lock(mtx_buffer);
    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    const double timestamp = get_time_sec(msg->header.stamp);

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    std::lock_guard<mutex> lock(mtx_buffer);
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        current_ring_field_present = lidar_ring_present_buffer.empty() ? false : lidar_ring_present_buffer.front();
        current_ring_valid_ratio = lidar_ring_ratio_buffer.empty() ? 0.0 : lidar_ring_ratio_buffer.front();
        current_ring_field_name = lidar_ring_name_buffer.empty() ? string() : lidar_ring_name_buffer.front();
        preprocess_time_last = lidar_preprocess_time_buffer.empty() ? 0.0 : lidar_preprocess_time_buffer.front();
        const double nominal_scantime =
            (p_pre != nullptr && p_pre->SCAN_RATE > 0) ? 1.0 / p_pre->SCAN_RATE : 0.1;
        // PointCloud2 data may be grouped by ring, so the last array element
        // is not guaranteed to be the latest point in the scan. FAST-LIO
        // sorts by curvature during deskew; use the same temporal definition
        // here to avoid shortening the LiDAR/IMU integration interval.
        double point_scantime = 0.0;
        for (const auto &point : meas.lidar->points)
        {
            const double offset_time = static_cast<double>(point.curvature) / 1000.0;
            if (std::isfinite(offset_time))
                point_scantime = std::max(point_scantime, offset_time);
        }
        const bool point_time_valid = std::isfinite(point_scantime) && point_scantime > 1e-6;

        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + nominal_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (!point_time_valid)
        {
            // Some KITTI PointCloud2 conversions do not provide per-point time.
            // Use the configured scan period so the IMU packet still spans a full scan.
            lidar_end_time = meas.lidar_beg_time + nominal_scantime;
        }
        else if (scan_num > 0 && point_scantime < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + point_scantime;
            lidar_mean_scantime += (point_scantime - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    if (!lidar_ring_present_buffer.empty()) lidar_ring_present_buffer.pop_front();
    if (!lidar_ring_ratio_buffer.empty()) lidar_ring_ratio_buffer.pop_front();
    if (!lidar_ring_name_buffer.empty()) lidar_ring_name_buffer.pop_front();
    if (!lidar_preprocess_time_buffer.empty()) lidar_preprocess_time_buffer.pop_front();
    lidar_gap_last = last_processed_lidar_time > 0.0 ?
                     meas.lidar_beg_time - last_processed_lidar_time : 0.0;
    last_processed_lidar_time = meas.lidar_beg_time;
    imu_samples_last = static_cast<int>(meas.imu.size());
    lidar_queue_size_last = lidar_buffer.size();
    imu_queue_size_last = imu_buffer.size();
    ++lidar_processed_count;
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
int infer_rank_ring(const PointType &p)
{
    if (rank_ring_source_mode != RANK_RING_ELEVATION)
    {
        if (rank_ring_source_mode == RANK_RING_POINTCLOUD2 &&
            !current_ring_field_present)
            return -1;

        const int preserved_ring = static_cast<int>(std::round(p.normal_x));
        return (std::isfinite(p.normal_x) && preserved_ring >= 0 && preserved_ring < p_pre->N_SCANS) ?
               preserved_ring : -1;
    }

    const double range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (range <= 1e-6) return -1;

    const double fov_total = rank_fov_up_deg - rank_fov_down_deg;
    if (fov_total <= 1e-6 || p_pre->N_SCANS <= 1) return -1;

    const double elev = std::asin(static_cast<double>(p.z) / range) * 180.0 / M_PI;
    const double row = (rank_fov_up_deg - elev) / fov_total * static_cast<double>(p_pre->N_SCANS - 1);
    const int ring = static_cast<int>(std::round(row));

    return (ring >= 0 && ring < p_pre->N_SCANS) ? ring : -1;
}

int compute_rank_col(const PointType &p, int cols)
{
    double azimuth_deg = std::atan2(static_cast<double>(p.y), static_cast<double>(p.x)) * 180.0 / M_PI;
    if (azimuth_deg < 0.0) azimuth_deg += 360.0;

    int col = static_cast<int>(std::round(azimuth_deg / rank_horizontal_resolution_deg));
    col %= cols;
    if (col < 0) col += cols;
    return col;
}

std::uint8_t to_u8(double value)
{
    value = clamp_value(value, 0.0, 1.0);
    return static_cast<std::uint8_t>(std::round(value * 255.0));
}

void publish_cylindrical_range_image(
    const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &pubRangeImage,
    const PointCloudXYZI::Ptr &cloud,
    const std::vector<float> &raw_ranks,
    const std::vector<bool> &raw_valid)
{
    if (!range_image_pub_en || !pubRangeImage || !cloud || cloud->empty())
        return;
    if (range_image_publish_every_n_frames > 1 &&
        scan_count % range_image_publish_every_n_frames != 0)
        return;
    if (pubRangeImage->get_subscription_count() == 0)
        return;

    const int raw_size = static_cast<int>(cloud->points.size());
    const int scan_lines = std::max(1, p_pre->N_SCANS);
    const int cols = std::max(1, static_cast<int>(std::round(
        360.0 / std::max(0.01, rank_horizontal_resolution_deg))));
    const int row_scale = std::max(1, range_image_row_scale);
    const double max_range = range_image_max_range > 0.0 ?
                             range_image_max_range :
                             std::max(1.0f, DET_RANGE);
    const bool has_rank =
        static_cast<int>(raw_ranks.size()) == raw_size &&
        static_cast<int>(raw_valid.size()) == raw_size;

    std::vector<float> cell_range(scan_lines * cols,
                                  std::numeric_limits<float>::infinity());
    std::vector<int> cell_index(scan_lines * cols, -1);

    for (int i = 0; i < raw_size; ++i)
    {
        if (has_rank && !raw_valid[i])
            continue;

        const PointType &p = cloud->points[i];
        const float range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (!std::isfinite(range) || range <= p_pre->blind || range > max_range)
            continue;

        const int ring = infer_rank_ring(p);
        if (ring < 0)
            continue;

        const int col = compute_rank_col(p, cols);
        const int cell = ring * cols + col;
        if (range < cell_range[cell])
        {
            cell_range[cell] = range;
            cell_index[cell] = i;
        }
    }

    double min_rank = std::numeric_limits<double>::infinity();
    double max_rank = -std::numeric_limits<double>::infinity();
    if (has_rank)
    {
        for (const int idx : cell_index)
        {
            if (idx < 0 || !std::isfinite(raw_ranks[idx]))
                continue;
            min_rank = std::min(min_rank, static_cast<double>(raw_ranks[idx]));
            max_rank = std::max(max_rank, static_cast<double>(raw_ranks[idx]));
        }
    }
    const bool draw_rank = has_rank && std::isfinite(min_rank) &&
                           std::isfinite(max_rank) && max_rank > min_rank;
    const double rank_denom = draw_rank ? (max_rank - min_rank) : 1.0;

    sensor_msgs::msg::Image image_msg;
    image_msg.header.stamp = get_ros_time(lidar_end_time);
    image_msg.header.frame_id = !lidar_frame_id.empty() ? lidar_frame_id :
                                (!body_frame_id.empty() ? body_frame_id : map_frame_id);
    image_msg.height = static_cast<std::uint32_t>(scan_lines * row_scale);
    image_msg.width = static_cast<std::uint32_t>(cols);
    image_msg.encoding = "rgb8";
    image_msg.is_bigendian = false;
    image_msg.step = image_msg.width * 3;
    image_msg.data.assign(image_msg.height * image_msg.step, 0);

    for (int ring = 0; ring < scan_lines; ++ring)
    {
        for (int col = 0; col < cols; ++col)
        {
            const int cell = ring * cols + col;
            const int idx = cell_index[cell];
            if (idx < 0)
                continue;

            const double range_norm = clamp_value(
                static_cast<double>(cell_range[cell]) / max_range, 0.0, 1.0);
            std::uint8_t r = 0;
            std::uint8_t g = 0;
            std::uint8_t b = 0;

            if (draw_rank)
            {
                const double rank_norm = clamp_value(
                    (static_cast<double>(raw_ranks[idx]) - min_rank) / rank_denom, 0.0, 1.0);
                r = to_u8(1.0 - rank_norm);
                g = to_u8(rank_norm);
                b = to_u8(0.25 + 0.75 * (1.0 - range_norm));
            }
            else
            {
                const std::uint8_t gray = to_u8(1.0 - range_norm);
                r = gray;
                g = gray;
                b = gray;
            }

            for (int dy = 0; dy < row_scale; ++dy)
            {
                const int out_row = ring * row_scale + dy;
                const size_t offset = (static_cast<size_t>(out_row) * cols + col) * 3;
                image_msg.data[offset + 0] = r;
                image_msg.data[offset + 1] = g;
                image_msg.data[offset + 2] = b;
            }
        }
    }

    pubRangeImage->publish(image_msg);
}

void publish_cylindrical_range_cloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubRangeCloud,
    const PointCloudXYZI::Ptr &cloud,
    const std::vector<float> &raw_ranks,
    const std::vector<bool> &raw_valid)
{
    if (!range_image_cloud_pub_en || !pubRangeCloud || !cloud || cloud->empty())
        return;
    if (range_image_publish_every_n_frames > 1 &&
        scan_count % range_image_publish_every_n_frames != 0)
        return;
    if (pubRangeCloud->get_subscription_count() == 0)
        return;

    const int raw_size = static_cast<int>(cloud->points.size());
    const int scan_lines = std::max(1, p_pre->N_SCANS);
    const int cols = std::max(1, static_cast<int>(std::round(
        360.0 / std::max(0.01, rank_horizontal_resolution_deg))));
    const double max_range = range_image_max_range > 0.0 ?
                             range_image_max_range :
                             std::max(1.0f, DET_RANGE);
    const bool has_rank =
        static_cast<int>(raw_ranks.size()) == raw_size &&
        static_cast<int>(raw_valid.size()) == raw_size;

    std::vector<float> cell_range(scan_lines * cols,
                                  std::numeric_limits<float>::infinity());
    std::vector<int> cell_index(scan_lines * cols, -1);

    for (int i = 0; i < raw_size; ++i)
    {
        if (has_rank && !raw_valid[i])
            continue;

        const PointType &p = cloud->points[i];
        const float range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (!std::isfinite(range) || range <= p_pre->blind || range > max_range)
            continue;

        const int ring = infer_rank_ring(p);
        if (ring < 0)
            continue;

        const int col = compute_rank_col(p, cols);
        const int cell = ring * cols + col;
        if (range < cell_range[cell])
        {
            cell_range[cell] = range;
            cell_index[cell] = i;
        }
    }

    double min_rank = std::numeric_limits<double>::infinity();
    double max_rank = -std::numeric_limits<double>::infinity();
    if (has_rank)
    {
        for (const int idx : cell_index)
        {
            if (idx < 0 || !std::isfinite(raw_ranks[idx]))
                continue;
            min_rank = std::min(min_rank, static_cast<double>(raw_ranks[idx]));
            max_rank = std::max(max_rank, static_cast<double>(raw_ranks[idx]));
        }
    }
    const bool draw_rank = has_rank && std::isfinite(min_rank) &&
                           std::isfinite(max_rank) && max_rank > min_rank;
    const double rank_denom = draw_rank ? (max_rank - min_rank) : 1.0;

    PointCloudXYZI::Ptr cylinder_cloud(new PointCloudXYZI());
    cylinder_cloud->reserve(static_cast<std::size_t>(std::min(raw_size, scan_lines * cols)));

    const double radius = std::max(0.1, range_image_cloud_radius);
    const double height = std::max(0.1, range_image_cloud_height);
    for (int ring = 0; ring < scan_lines; ++ring)
    {
        const double ring_ratio = scan_lines > 1 ?
                                  static_cast<double>(ring) /
                                  static_cast<double>(scan_lines - 1) : 0.5;
        for (int col = 0; col < cols; ++col)
        {
            const int cell = ring * cols + col;
            const int idx = cell_index[cell];
            if (idx < 0)
                continue;

            const double azimuth = 2.0 * M_PI * static_cast<double>(col) /
                                   static_cast<double>(cols);
            const double range_norm = clamp_value(
                static_cast<double>(cell_range[cell]) / max_range, 0.0, 1.0);

            double visual_rank = 1.0 + 3.0 * (1.0 - range_norm);
            if (draw_rank)
            {
                const double rank_norm = clamp_value(
                    (static_cast<double>(raw_ranks[idx]) - min_rank) / rank_denom, 0.0, 1.0);
                visual_rank = 1.0 + 3.0 * rank_norm;
            }

            PointType local_point;
            local_point.x = static_cast<float>(radius * std::cos(azimuth));
            local_point.y = static_cast<float>(radius * std::sin(azimuth));
            local_point.z = static_cast<float>(range_image_cloud_z_offset +
                                               (0.5 - ring_ratio) * height);
            local_point.intensity = static_cast<float>(visual_rank);
            local_point.normal_x = 0.0f;
            local_point.normal_y = 0.0f;
            local_point.normal_z = 0.0f;

            PointType world_point;
            world_point.normal_x = 0.0f;
            world_point.normal_y = 0.0f;
            world_point.normal_z = 0.0f;
            pointBodyToWorld(&local_point, &world_point);
            cylinder_cloud->push_back(world_point);
        }
    }

    if (cylinder_cloud->empty())
        return;

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cylinder_cloud, cloud_msg);
    cloud_msg.header.stamp = get_ros_time(lidar_end_time);
    cloud_msg.header.frame_id = map_frame_id;
    pubRangeCloud->publish(cloud_msg);
}

struct RankVoxelKey
{
    int64_t x;
    int64_t y;
    int64_t z;

    bool operator==(const RankVoxelKey &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct RankVoxelKeyHash
{
    std::size_t operator()(const RankVoxelKey &key) const
    {
        std::size_t h1 = std::hash<int64_t>()(key.x);
        std::size_t h2 = std::hash<int64_t>()(key.y);
        std::size_t h3 = std::hash<int64_t>()(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

RankVoxelKey compute_rank_voxel_key(const PointType &p, double voxel_size)
{
    return RankVoxelKey{
        static_cast<int64_t>(std::floor(p.x / voxel_size)),
        static_cast<int64_t>(std::floor(p.y / voxel_size)),
        static_cast<int64_t>(std::floor(p.z / voxel_size))
    };
}

void update_rank_valid_diagnostics(int raw_size)
{
    rank_valid_ratio_last = raw_size > 0 ?
                            static_cast<double>(rank_valid_num_last) / static_cast<double>(raw_size) : 0.0;
    if (rank_ring_source_mode == RANK_RING_POINTCLOUD2 &&
        (!current_ring_field_present ||
         current_ring_valid_ratio < rank_ring_valid_warning_ratio) &&
        scan_count % 20 == 0)
    {
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                    "[BOREAS_RT] ring preservation warning: field=%s present=%s preserved_ratio=%.3f "
                    "rank_valid_ratio=%.3f expected>=%.3f",
                    current_ring_field_name.empty() ? "missing" : current_ring_field_name.c_str(),
                    current_ring_field_present ? "true" : "false",
                    current_ring_valid_ratio,
                    rank_valid_ratio_last,
                    rank_ring_valid_warning_ratio);
    }
}

void compute_rank_for_cloud(const PointCloudXYZI::Ptr &cloud,
                            std::vector<float> &raw_ranks,
                            std::vector<bool> &raw_valid)
{
    const int raw_size = cloud ? static_cast<int>(cloud->points.size()) : 0;
    raw_ranks.assign(raw_size, 0.0f);
    raw_valid.assign(raw_size, false);
    rank_mean_last = 0.0;
    rank_valid_num_last = 0;
    rank_compute_time_last = 0.0;

    if (!rank_weight_enable || raw_size <= 0)
        return;
    const double rank_compute_start = omp_get_wtime();

    const int scan_lines = std::max(1, p_pre->N_SCANS);
    const int cols = std::max(1, static_cast<int>(std::round(360.0 / std::max(0.01, rank_horizontal_resolution_deg))));
    const double sigma_r = std::max(1e-6, rank_sigma_r);
    const double max_range = std::max(1.0f, DET_RANGE);

#ifdef FAST_LIO_USE_CUDA
    if (rank_gpu_enable && rank_gpu_active)
    {
        std::vector<RankCudaPoint> gpu_points(raw_size);
        for (int i = 0; i < raw_size; ++i)
        {
            const PointType &point = cloud->points[i];
            gpu_points[i] = RankCudaPoint{point.x, point.y, point.z, point.normal_x};
        }

        std::vector<std::uint8_t> gpu_valid(raw_size, 0);
        int gpu_valid_count = 0;
        float gpu_rank_sum = 0.0f;
        std::string gpu_error;
        const bool gpu_computed = rank_cuda_compute(gpu_points.data(),
                                                     raw_size,
                                                     scan_lines,
                                                     cols,
                                                     static_cast<float>(rank_horizontal_resolution_deg),
                                                     static_cast<float>(sigma_r),
                                                     static_cast<float>(p_pre->blind),
                                                     static_cast<float>(max_range),
                                                     static_cast<float>(rank_fov_up_deg),
                                                     static_cast<float>(rank_fov_down_deg),
                                                     rank_ring_source_mode,
                                                     current_ring_field_present,
                                                     rank_range_factor_mode_id,
                                                     raw_ranks.data(),
                                                     gpu_valid.data(),
                                                     &gpu_valid_count,
                                                     &gpu_rank_sum,
                                                     &gpu_error);
        if (gpu_computed)
        {
            for (int i = 0; i < raw_size; ++i)
                raw_valid[i] = gpu_valid[i] != 0;
            rank_valid_num_last = gpu_valid_count;
            rank_mean_last = gpu_rank_sum / std::max(1, gpu_valid_count);
            rank_compute_time_last = omp_get_wtime() - rank_compute_start;
            update_rank_valid_diagnostics(raw_size);
            return;
        }

        rank_gpu_active = false;
        RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                    "[GPU] full rank pipeline failed; switching to CPU: %s",
                    gpu_error.c_str());
    }
#endif

    std::vector<float> ranges(raw_size, 0.0f);
    std::vector<int> rings(raw_size, -1);
    std::vector<int> cols_idx(raw_size, -1);
    std::vector<int> valid_indices;
    valid_indices.reserve(raw_size);

    for (int i = 0; i < raw_size; ++i)
    {
        const PointType &p = cloud->points[i];
        const float range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        ranges[i] = range;
        if (!std::isfinite(range) || range <= p_pre->blind || range > DET_RANGE)
            continue;

        const int ring = infer_rank_ring(p);
        if (ring < 0)
            continue;

        rings[i] = ring;
        cols_idx[i] = compute_rank_col(p, cols);
        raw_valid[i] = true;
        valid_indices.push_back(i);
    }

    if (valid_indices.empty())
    {
        rank_compute_time_last = omp_get_wtime() - rank_compute_start;
        update_rank_valid_diagnostics(raw_size);
        return;
    }

    std::vector<int> range_image(scan_lines * cols, -1);
    for (const int idx : valid_indices)
    {
        const int cell = rings[idx] * cols + cols_idx[idx];
        const int old_idx = range_image[cell];
        if (old_idx < 0 || ranges[idx] < ranges[old_idx])
            range_image[cell] = idx;
    }

    constexpr int kernel_radius = 2;
    constexpr double neighbor_total = 25.0;
    for (const int idx : valid_indices)
    {
        double consistency_sum = 0.0;

        for (int du = -kernel_radius; du <= kernel_radius; ++du)
        {
            const int nr = rings[idx] + du;
            if (nr < 0 || nr >= scan_lines)
                continue;

            for (int dv = -kernel_radius; dv <= kernel_radius; ++dv)
            {
                int nc = cols_idx[idx] + dv;
                if (nc < 0) nc += cols;
                if (nc >= cols) nc -= cols;

                const int neighbor_idx = range_image[nr * cols + nc];
                if (neighbor_idx < 0)
                    continue;

                const double dr = static_cast<double>(ranges[idx] - ranges[neighbor_idx]);
                consistency_sum += std::exp(-(dr * dr) / (2.0 * sigma_r * sigma_r));
            }
        }

        const double consistency = consistency_sum / neighbor_total;
        const double normalized_range = static_cast<double>(ranges[idx]) / max_range;
        double range_factor = 1.0;
        if (rank_range_factor_mode_id == 0)
            range_factor += normalized_range;
        else if (rank_range_factor_mode_id == 2)
            range_factor = 1.0 / (1.0 + normalized_range);
        raw_ranks[idx] = static_cast<float>((1.0 + consistency) * range_factor);
    }

    double sum_rank = 0.0;
    for (const int idx : valid_indices)
        sum_rank += std::max(static_cast<double>(raw_ranks[idx]), 1e-6);

    rank_valid_num_last = static_cast<int>(valid_indices.size());
    rank_mean_last = sum_rank / std::max(1, rank_valid_num_last);
    rank_compute_time_last = omp_get_wtime() - rank_compute_start;
    update_rank_valid_diagnostics(raw_size);
}

void print_rank_terminal_distribution(const PointCloudXYZI::Ptr &cloud,
                                      const std::vector<float> &raw_ranks,
                                      const std::vector<bool> &raw_valid)
{
    if (!rank_terminal_distribution_enable ||
        rank_terminal_distribution_every_n_frames <= 0 ||
        scan_count % rank_terminal_distribution_every_n_frames != 0 ||
        !cloud || cloud->empty())
        return;

    const double max_range = rank_terminal_distribution_max_range > 0.0 ?
                             rank_terminal_distribution_max_range :
                             static_cast<double>(DET_RANGE);
    const double range_bin_size = std::max(0.1, rank_terminal_distribution_range_bin_size);
    const double az_bin_deg = std::max(1.0, rank_terminal_distribution_azimuth_bin_deg);
    const int range_bins = std::max(1, std::min(60, static_cast<int>(std::ceil(max_range / range_bin_size))));
    const int az_bins = std::max(1, std::min(120, static_cast<int>(std::ceil(360.0 / az_bin_deg))));

    std::vector<double> rank_sum(range_bins * az_bins, 0.0);
    std::vector<int> count(range_bins * az_bins, 0);
    int total_count = 0;

    const int raw_size = std::min({static_cast<int>(cloud->points.size()),
                                   static_cast<int>(raw_ranks.size()),
                                   static_cast<int>(raw_valid.size())});
    for (int i = 0; i < raw_size; ++i)
    {
        if (!raw_valid[i] || !std::isfinite(raw_ranks[i]))
            continue;

        const PointType &p = cloud->points[i];
        const double range = std::sqrt(p.x * p.x + p.y * p.y);
        if (!std::isfinite(range) || range <= 0.0 || range > max_range)
            continue;

        double azimuth_deg = std::atan2(p.y, p.x) * 180.0 / PI_M;
        if (azimuth_deg < -180.0)
            azimuth_deg += 360.0;
        if (azimuth_deg >= 180.0)
            azimuth_deg -= 360.0;

        const int r = std::min(range_bins - 1, static_cast<int>(range / range_bin_size));
        const int a = std::min(az_bins - 1,
                               std::max(0, static_cast<int>((azimuth_deg + 180.0) / (360.0 / az_bins))));
        const int cell = r * az_bins + a;
        rank_sum[cell] += raw_ranks[i];
        count[cell] += 1;
        ++total_count;
    }

    if (total_count == 0)
        return;

    double min_avg = std::numeric_limits<double>::max();
    double max_avg = std::numeric_limits<double>::lowest();
    for (int i = 0; i < range_bins * az_bins; ++i)
    {
        if (count[i] <= 0)
            continue;
        const double avg = rank_sum[i] / static_cast<double>(count[i]);
        min_avg = std::min(min_avg, avg);
        max_avg = std::max(max_avg, avg);
    }

    static const std::string palette = " .:-=+*#%@";
    std::ostringstream oss;
    oss << "\n[RANK_DIST] cylindrical 2D distribution avg_rank, range x azimuth"
        << " points=" << total_count
        << " range_bin=" << std::fixed << std::setprecision(1) << range_bin_size << "m"
        << " az_bin=" << std::setprecision(1) << (360.0 / az_bins) << "deg"
        << " rank=[" << std::setprecision(3) << min_avg << "," << max_avg << "]\n"
        << "azimuth -180" << std::setw(std::max(1, az_bins - 14)) << "" << "+180\n";

    const double denom = std::max(1e-6, max_avg - min_avg);
    for (int r = range_bins - 1; r >= 0; --r)
    {
        oss << std::setw(4) << static_cast<int>(r * range_bin_size) << "-"
            << std::setw(4) << static_cast<int>(std::min(max_range, (r + 1) * range_bin_size)) << "m |";
        for (int a = 0; a < az_bins; ++a)
        {
            const int cell = r * az_bins + a;
            if (count[cell] <= 0)
            {
                oss << ' ';
                continue;
            }

            const double avg = rank_sum[cell] / static_cast<double>(count[cell]);
            const int palette_idx = std::min(static_cast<int>(palette.size()) - 1,
                                             std::max(0, static_cast<int>(std::round((avg - min_avg) / denom *
                                                                                    (palette.size() - 1)))));
            oss << palette[palette_idx];
        }
        oss << "|\n";
    }

    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "%s", oss.str().c_str());
}

void compute_weight_from_downsampled_rank()
{
    point_weights.assign(feats_down_size, static_cast<float>(rank_min_weight));

    double sum_rank = 0.0;
    int valid_count = 0;
    for (int i = 0; i < feats_down_size; ++i)
    {
        if (i >= static_cast<int>(point_rank_valid.size()) || !point_rank_valid[i])
            continue;
        sum_rank += std::max(static_cast<double>(point_ranks[i]), 1e-6);
        ++valid_count;
    }

    if (valid_count == 0)
        return;

    const double n = static_cast<double>(valid_count);
    for (int i = 0; i < feats_down_size; ++i)
    {
        if (i >= static_cast<int>(point_rank_valid.size()) || !point_rank_valid[i])
            continue;

        double weight = n * std::max(static_cast<double>(point_ranks[i]), 1e-6) / (sum_rank + 1e-6);
        weight = clamp_value(weight, rank_min_weight, rank_max_weight);
        point_weights[i] = static_cast<float>(weight);
    }
}

void rank_aware_downsample(
    const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &pubRangeImage,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubRangeCloud)
{
    rank_valid_num_last = 0;
    rank_valid_ratio_last = 0.0;
    rank_mean_last = 0.0;
    point_ranks.clear();
    point_weights.clear();
    point_rank_valid.clear();
    effective_point_weights.clear();
    effective_point_ranks.clear();
    latest_raw_point_ranks.clear();
    latest_raw_point_rank_valid.clear();
    frame_rank_weight = 1.0;
    rank_map_threshold_last = 0.0f;

    if (!rank_weight_enable)
    {
        std::vector<float> raw_ranks;
        std::vector<bool> raw_valid;
        publish_cylindrical_range_image(pubRangeImage, feats_undistort, raw_ranks, raw_valid);
        publish_cylindrical_range_cloud(pubRangeCloud, feats_undistort, raw_ranks, raw_valid);
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        feats_down_size = feats_down_body->points.size();
        point_ranks.assign(feats_down_size, 0.0f);
        point_weights.assign(feats_down_size, 1.0f);
        point_rank_valid.assign(feats_down_size, false);
        return;
    }

    std::vector<float> raw_ranks;
    std::vector<bool> raw_valid;
    compute_rank_for_cloud(feats_undistort, raw_ranks, raw_valid);
    latest_raw_point_ranks = raw_ranks;
    latest_raw_point_rank_valid = raw_valid;
    publish_cylindrical_range_image(pubRangeImage, feats_undistort, raw_ranks, raw_valid);
    publish_cylindrical_range_cloud(pubRangeCloud, feats_undistort, raw_ranks, raw_valid);
    print_rank_terminal_distribution(feats_undistort, raw_ranks, raw_valid);

    const int raw_size = feats_undistort ? static_cast<int>(feats_undistort->points.size()) : 0;
    if (raw_size <= 0 || rank_valid_num_last == 0)
    {
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        feats_down_size = 0;
        feats_down_size = feats_down_body->points.size();
        point_ranks.assign(feats_down_size, 0.0f);
        point_weights.assign(feats_down_size, static_cast<float>(rank_min_weight));
        point_rank_valid.assign(feats_down_size, false);
        return;
    }

    const double voxel_size = std::max(1e-6, filter_size_surf_min);
    std::unordered_map<RankVoxelKey, int, RankVoxelKeyHash> best_idx;
    best_idx.reserve(static_cast<size_t>(rank_valid_num_last));

    for (int i = 0; i < raw_size; ++i)
    {
        if (!raw_valid[i])
            continue;

        const RankVoxelKey key = compute_rank_voxel_key(feats_undistort->points[i], voxel_size);
        auto it = best_idx.find(key);
        if (it == best_idx.end() || raw_ranks[i] > raw_ranks[it->second])
            best_idx[key] = i;
    }

    std::vector<int> selected_indices;
    selected_indices.reserve(best_idx.size());
    for (const auto &kv : best_idx)
        selected_indices.push_back(kv.second);
    std::sort(selected_indices.begin(), selected_indices.end());

    feats_down_body->clear();
    feats_down_body->reserve(selected_indices.size());
    point_ranks.reserve(selected_indices.size());
    point_rank_valid.reserve(selected_indices.size());

    for (const int idx : selected_indices)
    {
        feats_down_body->push_back(feats_undistort->points[idx]);
        point_ranks.push_back(raw_ranks[idx]);
        point_rank_valid.push_back(raw_valid[idx]);
    }

    feats_down_size = feats_down_body->points.size();
    compute_weight_from_downsampled_rank();
}

float rank_map_threshold(double percent_override = -1.0)
{
    if (!rank_weight_enable || !rank_map_filter_enable || rank_valid_num_last <= 0 ||
        point_ranks.empty() || rank_map_filter_percent <= 0.0)
        return -std::numeric_limits<float>::infinity();

    std::vector<float> sorted;
    sorted.reserve(point_ranks.size());
    for (size_t i = 0; i < point_ranks.size(); ++i)
    {
        if (i < point_rank_valid.size() && point_rank_valid[i] && std::isfinite(point_ranks[i]))
            sorted.push_back(point_ranks[i]);
    }

    if (sorted.empty())
        return -std::numeric_limits<float>::infinity();

    const double requested_percent = percent_override >= 0.0 ?
                                     percent_override : rank_map_filter_percent;
    const double percent = clamp_value(requested_percent, 0.0, 90.0);
    const size_t nth = std::min(sorted.size() - 1,
                                static_cast<size_t>(std::floor((percent / 100.0) *
                                                               static_cast<double>(sorted.size()))));
    std::nth_element(sorted.begin(), sorted.begin() + nth, sorted.end());
    rank_map_threshold_last = sorted[nth];
    return rank_map_threshold_last;
}

void map_incremental()
{
    add_point_size = 0;
    if (map_confidence_fail_safe_enable &&
        effct_feat_num < map_skip_below_effective_points)
    {
        return;
    }

    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    const bool low_support = map_confidence_fail_safe_enable &&
                             (effct_feat_num < map_min_effective_points ||
                              rank_valid_ratio_last < map_min_rank_valid_ratio);
    const double map_filter_percent = low_support ?
                                      rank_map_filter_percent * map_threshold_relax_factor :
                                      rank_map_filter_percent;
    const float map_rank_thr = rank_map_threshold(map_filter_percent);
    for (int i = 0; i < feats_down_size; i++)
    {
        if (rank_weight_enable && rank_map_filter_enable && rank_valid_num_last > 0)
        {
            const bool rank_valid = i < static_cast<int>(point_rank_valid.size()) && point_rank_valid[i];
            const float rank = i < static_cast<int>(point_ranks.size()) ? point_ranks[i] : 0.0f;
            if (!rank_valid || rank < map_rank_thr)
                continue;
        }

        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = map_frame_id;
        pubLaserCloudFull->publish(laserCloudmsg);
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/

    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;
    }
    //     static int scan_wait_num = 0;
    //     scan_wait_num ++;
    //     if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
    //     {
    //         pcd_index ++;
    //         string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
    //         pcl::PCDWriter pcd_writer;
    //         cout << "current scan saved to /PCD/" << all_points_dir << endl;
    //         pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    //         pcl_wait_save->clear();
    //         scan_wait_num = 0;
    //     }
    // }

}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = body_frame_id;
    pubLaserCloudFull_body->publish(laserCloudmsg);
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = map_frame_id;
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_weather_filtered_world(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubWeatherFilteredCloud)
{
    if (!filtered_cloud_pub_en || !pubWeatherFilteredCloud ||
        !feats_down_body || feats_down_body->empty())
        return;
    if (pubWeatherFilteredCloud->get_subscription_count() == 0)
        return;

    const float map_rank_thr =
        (rank_weight_enable && rank_map_filter_enable && rank_valid_num_last > 0) ?
        rank_map_threshold() : -std::numeric_limits<float>::infinity();

    PointCloudXYZI::Ptr filtered_world(new PointCloudXYZI());
    filtered_world->reserve(feats_down_body->points.size());

    for (int i = 0; i < feats_down_size; ++i)
    {
        if (i >= static_cast<int>(feats_down_body->points.size()))
            break;

        if (rank_weight_enable && rank_map_filter_enable && rank_valid_num_last > 0)
        {
            const bool rank_valid = i < static_cast<int>(point_rank_valid.size()) && point_rank_valid[i];
            const float rank = i < static_cast<int>(point_ranks.size()) ? point_ranks[i] : 0.0f;
            if (!rank_valid || rank < map_rank_thr)
                continue;
        }

        PointType world_point;
        RGBpointBodyToWorld(&feats_down_body->points[i], &world_point);
        if (i < static_cast<int>(point_rank_valid.size()) && point_rank_valid[i] &&
            i < static_cast<int>(point_ranks.size()) && std::isfinite(point_ranks[i]))
        {
            world_point.intensity = point_ranks[i];
        }
        filtered_world->push_back(world_point);
    }

    sensor_msgs::msg::PointCloud2 filtered_msg;
    pcl::toROSMsg(*filtered_world, filtered_msg);
    filtered_msg.header.stamp = get_ros_time(lidar_end_time);
    filtered_msg.header.frame_id = map_frame_id;
    pubWeatherFilteredCloud->publish(filtered_msg);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    if (!map_pub_en || !feats_down_body || feats_down_body->empty())
        return;

    PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                            &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub += *laserCloudWorld;

    if (static_cast<int>(pcl_wait_pub->size()) > map_visual_max_points)
    {
        pcl::VoxelGrid<PointType> visual_filter;
        visual_filter.setLeafSize(map_visual_voxel_size,
                                  map_visual_voxel_size,
                                  map_visual_voxel_size);
        visual_filter.setInputCloud(pcl_wait_pub);
        PointCloudXYZI::Ptr filtered(new PointCloudXYZI());
        visual_filter.filter(*filtered);

        if (static_cast<int>(filtered->size()) > map_visual_max_points)
        {
            PointCloudXYZI::Ptr capped(new PointCloudXYZI());
            capped->reserve(map_visual_max_points);
            const double stride = static_cast<double>(filtered->size()) /
                                  static_cast<double>(map_visual_max_points);
            for (int i = 0; i < map_visual_max_points; ++i)
                capped->push_back(filtered->points[static_cast<std::size_t>(i * stride)]);
            pcl_wait_pub.swap(capped);
        }
        else
        {
            pcl_wait_pub.swap(filtered);
        }
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = map_frame_id;
    pubLaserCloudMap->publish(laserCloudmsg);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;

}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = map_frame_id;
    odomAftMapped.child_frame_id = body_frame_id;
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped->publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = map_frame_id;
    trans.header.stamp = odomAftMapped.header.stamp;
    trans.child_frame_id = body_frame_id;
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time);
    msg_body_pose.header.frame_id = map_frame_id;

    // Save every processed LiDAR frame for trajectory evaluation.
    path.poses.push_back(msg_body_pose);

    // Publish to RViz only every 10 frames to avoid visualization overload.
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    effective_point_weights.clear();
    effective_point_ranks.clear();
    total_residual = 0.0;
    double nearest_search_time = 0.0;

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for reduction(+:nearest_search_time)
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            const double search_begin = omp_get_wtime();
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            nearest_search_time += omp_get_wtime() - search_begin;
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    kdtree_search_time += nearest_search_time;

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            const float rank_w = i < static_cast<int>(point_weights.size()) ? point_weights[i] : 1.0f;
            const float raw_rank = i < static_cast<int>(point_ranks.size()) ? point_ranks[i] : 0.0f;
            effective_point_weights.push_back(rank_w);
            effective_point_ranks.push_back(raw_rank);
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        lidar_confidence_last = 0.0;
        lidar_degradation_last = 1.0;
        frame_rank_weight = rank_frame_confidence_enable ?
                            1.0 / (1.0 + rank_frame_gamma) : 1.0;
        adaptive_lidar_cov = adaptive_lidar_cov_enable ? adaptive_cov_max : base_lidar_cov;
        ekfom_data.valid = false;
        std::cerr << "No Effective Points! feats_down_size=" << feats_down_size
                  << " map_points=" << ikdtree.validnum()
                  << " rank_valid=" << rank_valid_num_last << std::endl;
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    {
        // c_k is a frame-level LiDAR confidence. It is computed independently
        // from whether frame weighting or adaptive covariance is enabled so
        // baseline and proposed runs log the same degradation signal.
        residual_variance_last = 0.0;
        double mean_rank = 0.0;
        int valid_rank_num = 0;
        for (int i = 0; i < effct_feat_num; ++i)
        {
            const double residual = std::fabs(static_cast<double>(corr_normvect->points[i].intensity));
            const double diff = residual - res_mean_last;
            residual_variance_last += diff * diff;
            if (i < static_cast<int>(effective_point_ranks.size()) && effective_point_ranks[i] > 0.0f)
            {
                mean_rank += effective_point_ranks[i];
                ++valid_rank_num;
            }
        }
        residual_variance_last /= std::max(1, effct_feat_num);
        mean_rank /= std::max(1, valid_rank_num);

        const bool rank_signal_available = rank_weight_enable && valid_rank_num > 0;
        const double q_rank = rank_signal_available ?
                              clamp_value(mean_rank / std::max(1e-6, rank_frame_ref_mean), 0.0, 1.0) : 0.0;
        const double q_count = clamp_value(static_cast<double>(effct_feat_num) /
                                          std::max(1.0, rank_frame_ref_points), 0.0, 1.0);
        const double sigma_e = std::max(1e-6, rank_frame_residual_sigma);
        const double q_residual = std::exp(-residual_variance_last / (sigma_e * sigma_e));
        // Normalize only over signals that exist. A FAST-LIO2 baseline with
        // ranking disabled must not be classified as degraded merely because
        // q_rank is unavailable.
        const double active_rank_alpha = rank_signal_available ? rank_frame_alpha_rank : 0.0;
        const double alpha_sum = std::max(1e-6,
                                          active_rank_alpha +
                                          rank_frame_alpha_count +
                                          rank_frame_alpha_residual);
        lidar_confidence_last = clamp_value((active_rank_alpha * q_rank +
                                             rank_frame_alpha_count * q_count +
                                             rank_frame_alpha_residual * q_residual) / alpha_sum,
                                            0.0, 1.0);
        lidar_degradation_last = 1.0 - lidar_confidence_last;
        frame_rank_weight = rank_frame_confidence_enable ?
                            1.0 / (1.0 + rank_frame_gamma * lidar_degradation_last) : 1.0;

        if (adaptive_lidar_cov_enable)
        {
            if (adaptive_cov_mode == "confidence")
            {
                adaptive_lidar_cov = base_lidar_cov /
                                     clamp_value(lidar_confidence_last,
                                                 adaptive_cov_min_confidence,
                                                 1.0);
            }
            else
            {
                adaptive_lidar_cov = base_lidar_cov *
                                     (1.0 + adaptive_cov_degradation_gamma * lidar_degradation_last);
            }
            adaptive_lidar_cov = clamp_value(adaptive_lidar_cov,
                                             adaptive_cov_min,
                                             adaptive_cov_max);
        }
        else
        {
            adaptive_lidar_cov = base_lidar_cov;
        }
    }

    /*
     * This remains a confidence-weighted LiDAR measurement update, not an
     * explicit per-point covariance update. Rank, Huber, and frame weights are
     * combined below and sqrt(total_w) scales both H and h. The separate
     * adaptive_lidar_cov above is the explicit frame-level R_k^LiDAR used by
     * the IEKF update.
     */
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measurement weighting.
         * total_w behaves like an inverse measurement covariance multiplier.
         * LASER_POINT_COV is still global; per-point rank/Huber/frame confidence
         * are applied by scaling both H and the residual before the IEKF update.
         */
        double rank_w = i < static_cast<int>(effective_point_weights.size()) ? effective_point_weights[i] : 1.0;
        double huber_w = 1.0;
        if (rank_huber_enable)
        {
            const double abs_e = std::fabs(static_cast<double>(norm_p.intensity));
            if (abs_e > rank_huber_delta && abs_e > 1e-9)
                huber_w = rank_huber_delta / abs_e;
        }

        const double total_w = std::max(1e-6, frame_rank_weight * rank_w * huber_w);
        const double sqrt_w = std::sqrt(total_w);
        ekfom_data.h_x.row(i) *= sqrt_w;
        ekfom_data.h(i) = -sqrt_w * norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

/*
* @brief : Save the whole trajectory to a txt file (TUM format)
*/
void save_trajectory(const std::string &traj_file) {
    std::string filename(traj_file);
    std::fstream output_fstream;

    output_fstream.open(filename, std::ios_base::out);

    if (!output_fstream.is_open()) {
        std::cerr << "Failed to open " << filename << '\n';
    }

    else {
        output_fstream << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
        for (const auto &p : path.poses) {
            output_fstream << std::setprecision(15) << p.header.stamp.sec + p.header.stamp.nanosec * 1e-9 << " "
                           << p.pose.position.x << " "
                           << p.pose.position.y << " "
                           << p.pose.position.z << " "
                           << p.pose.orientation.x << " "
                           << p.pose.orientation.y << " "
                           << p.pose.orientation.z << " "
                           << p.pose.orientation.w << std::endl;
        }
    }
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<bool>("publish.map_publish_en", true);
        this->declare_parameter<double>("publish.map_publish_period_sec", 3.0);
        this->declare_parameter<double>("publish.map_visual_voxel_size", 0.75);
        this->declare_parameter<int>("publish.map_visual_max_points", 250000);
        this->declare_parameter<bool>("publish.range_image_en", true);
        this->declare_parameter<string>("publish.range_image_topic", "/debug/cylindrical_range_image");
        this->declare_parameter<bool>("publish.range_image_cloud_en", true);
        this->declare_parameter<string>("publish.range_image_cloud_topic", "/debug/cylindrical_range_cloud");
        this->declare_parameter<bool>("publish.filtered_cloud_en", true);
        this->declare_parameter<string>("publish.filtered_cloud_topic", "/cloud_weather_filtered");
        this->declare_parameter<int>("publish.range_image_every_n_frames", 1);
        this->declare_parameter<int>("publish.range_image_row_scale", 8);
        this->declare_parameter<double>("publish.range_image_max_range", 0.0);
        this->declare_parameter<double>("publish.range_image_cloud_radius", 15.0);
        this->declare_parameter<double>("publish.range_image_cloud_height", 5.0);
        this->declare_parameter<double>("publish.range_image_cloud_z_offset", 0.0);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<string>("common.lid_frame_id", "");
        this->declare_parameter<string>("common.imu_frame_id", "");
        this->declare_parameter<string>("common.map_frame_id", "camera_init");
        this->declare_parameter<string>("common.body_frame_id", "body");
        this->declare_parameter<string>("dataset.name", "generic");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<int>("common.lidar_qos_depth", 50);
        this->declare_parameter<int>("common.imu_qos_depth", 5000);
        this->declare_parameter<bool>("common.lidar_qos_reliable", false);
        this->declare_parameter<bool>("common.imu_qos_reliable", false);
        this->declare_parameter<int>("common.executor_threads", 4);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<bool>("mapping.init_gyro_bias_from_mean", true);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());
        this->declare_parameter<bool>("traj_save.traj_save_en", false);
        this->declare_parameter<string>("traj_save.traj_file_path", "");
        this->declare_parameter<bool>("rank_weight.enable", true);
        this->declare_parameter<bool>("rank_weight.gpu_enable", true);
        this->declare_parameter<string>("rank_weight.ring_source", "normal_x");
        this->declare_parameter<double>("rank_weight.ring_valid_warning_ratio", 0.9);
        this->declare_parameter<double>("rank_weight.horizontal_resolution_deg", 0.2);
        this->declare_parameter<double>("rank_weight.sigma_r", 1.0);
        this->declare_parameter<string>("rank_weight.range_factor_mode", "linear_boost");
        this->declare_parameter<double>("rank_weight.min_weight", 0.2);
        this->declare_parameter<double>("rank_weight.max_weight", 2.0);
        this->declare_parameter<bool>("rank_weight.huber_enable", false);
        this->declare_parameter<double>("rank_weight.huber_delta", 0.2);
        this->declare_parameter<bool>("rank_weight.map_filter_enable", true);
        this->declare_parameter<double>("rank_weight.map_filter_percent", 20.0);
        this->declare_parameter<double>("rank_weight.fov_up_deg", 2.0);
        this->declare_parameter<double>("rank_weight.fov_down_deg", -24.8);
        this->declare_parameter<bool>("rank_weight.frame_confidence_enable", false);
        this->declare_parameter<double>("rank_weight.frame_alpha_rank", 0.5);
        this->declare_parameter<double>("rank_weight.frame_alpha_count", 0.3);
        this->declare_parameter<double>("rank_weight.frame_alpha_residual", 0.2);
        this->declare_parameter<double>("rank_weight.frame_gamma", 2.0);
        this->declare_parameter<double>("rank_weight.frame_ref_points", 1000.0);
        this->declare_parameter<double>("rank_weight.frame_residual_sigma", 0.1);
        this->declare_parameter<double>("rank_weight.frame_ref_mean", 1.5);
        this->declare_parameter<int>("rank_weight.log_every_n_frames", 20);
        this->declare_parameter<bool>("rank_weight.terminal_distribution_enable", false);
        this->declare_parameter<int>("rank_weight.terminal_distribution_every_n_frames", 20);
        this->declare_parameter<double>("rank_weight.terminal_distribution_range_bin_size", 10.0);
        this->declare_parameter<double>("rank_weight.terminal_distribution_azimuth_bin_deg", 15.0);
        this->declare_parameter<double>("rank_weight.terminal_distribution_max_range", 0.0);
        this->declare_parameter<bool>("adaptive_lidar_cov.enable", false);
        this->declare_parameter<string>("adaptive_lidar_cov.mode", "degradation");
        this->declare_parameter<double>("adaptive_lidar_cov.base_cov", LASER_POINT_COV);
        this->declare_parameter<double>("adaptive_lidar_cov.min_confidence", 0.1);
        this->declare_parameter<double>("adaptive_lidar_cov.degradation_gamma", 4.0);
        this->declare_parameter<double>("adaptive_lidar_cov.min_cov", LASER_POINT_COV);
        this->declare_parameter<double>("adaptive_lidar_cov.max_cov", 0.1);
        this->declare_parameter<bool>("map_confidence_fail_safe.enable", true);
        this->declare_parameter<int>("map_confidence_fail_safe.min_effective_points", 100);
        this->declare_parameter<int>("map_confidence_fail_safe.skip_below_effective_points", 20);
        this->declare_parameter<double>("map_confidence_fail_safe.min_rank_valid_ratio", 0.2);
        this->declare_parameter<double>("map_confidence_fail_safe.threshold_relax_factor", 0.5);
        this->declare_parameter<bool>("experiment_log.enable", true);
        this->declare_parameter<string>("experiment_log.path", "");

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<bool>("publish.map_publish_en", map_pub_en, true);
        this->get_parameter_or<double>("publish.map_publish_period_sec", map_publish_period_sec, 3.0);
        this->get_parameter_or<double>("publish.map_visual_voxel_size", map_visual_voxel_size, 0.75);
        this->get_parameter_or<int>("publish.map_visual_max_points", map_visual_max_points, 250000);
        this->get_parameter_or<bool>("publish.range_image_en", range_image_pub_en, true);
        this->get_parameter_or<string>("publish.range_image_topic", range_image_topic, "/debug/cylindrical_range_image");
        this->get_parameter_or<bool>("publish.range_image_cloud_en", range_image_cloud_pub_en, true);
        this->get_parameter_or<string>("publish.range_image_cloud_topic", range_image_cloud_topic, "/debug/cylindrical_range_cloud");
        this->get_parameter_or<bool>("publish.filtered_cloud_en", filtered_cloud_pub_en, true);
        this->get_parameter_or<string>("publish.filtered_cloud_topic", filtered_cloud_topic, "/cloud_weather_filtered");
        this->get_parameter_or<int>("publish.range_image_every_n_frames", range_image_publish_every_n_frames, 1);
        this->get_parameter_or<int>("publish.range_image_row_scale", range_image_row_scale, 8);
        this->get_parameter_or<double>("publish.range_image_max_range", range_image_max_range, 0.0);
        this->get_parameter_or<double>("publish.range_image_cloud_radius", range_image_cloud_radius, 15.0);
        this->get_parameter_or<double>("publish.range_image_cloud_height", range_image_cloud_height, 5.0);
        this->get_parameter_or<double>("publish.range_image_cloud_z_offset", range_image_cloud_z_offset, 0.0);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<string>("common.lid_frame_id", lidar_frame_id, "");
        this->get_parameter_or<string>("common.imu_frame_id", imu_frame_id, "");
        this->get_parameter_or<string>("common.map_frame_id", map_frame_id, "camera_init");
        this->get_parameter_or<string>("common.body_frame_id", body_frame_id, "body");
        this->get_parameter_or<string>("dataset.name", dataset_name, "generic");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<int>("common.lidar_qos_depth", lidar_qos_depth, 50);
        this->get_parameter_or<int>("common.imu_qos_depth", imu_qos_depth, 5000);
        this->get_parameter_or<bool>("common.lidar_qos_reliable", lidar_qos_reliable, false);
        this->get_parameter_or<bool>("common.imu_qos_reliable", imu_qos_reliable, false);
        this->get_parameter_or<int>("common.executor_threads", executor_threads, 4);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<bool>("mapping.init_gyro_bias_from_mean", init_gyro_bias_from_mean, true);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        this->get_parameter_or<bool>("traj_save.traj_save_en", traj_save_en, false);
        this->get_parameter_or<string>("traj_save.traj_file_path", traj_file_path, "");
        this->get_parameter_or<bool>("rank_weight.enable", rank_weight_enable, true);
        this->get_parameter_or<bool>("rank_weight.gpu_enable", rank_gpu_enable, true);
        this->get_parameter_or<string>("rank_weight.ring_source", rank_ring_source, "normal_x");
        this->get_parameter_or<double>("rank_weight.ring_valid_warning_ratio", rank_ring_valid_warning_ratio, 0.9);
        this->get_parameter_or<double>("rank_weight.horizontal_resolution_deg", rank_horizontal_resolution_deg, 0.2);
        this->get_parameter_or<double>("rank_weight.sigma_r", rank_sigma_r, 1.0);
        this->get_parameter_or<string>("rank_weight.range_factor_mode", rank_range_factor_mode, "linear_boost");
        this->get_parameter_or<double>("rank_weight.min_weight", rank_min_weight, 0.2);
        this->get_parameter_or<double>("rank_weight.max_weight", rank_max_weight, 2.0);
        this->get_parameter_or<bool>("rank_weight.huber_enable", rank_huber_enable, false);
        this->get_parameter_or<double>("rank_weight.huber_delta", rank_huber_delta, 0.2);
        this->get_parameter_or<bool>("rank_weight.map_filter_enable", rank_map_filter_enable, true);
        this->get_parameter_or<double>("rank_weight.map_filter_percent", rank_map_filter_percent, 20.0);
        this->get_parameter_or<double>("rank_weight.fov_up_deg", rank_fov_up_deg, 2.0);
        this->get_parameter_or<double>("rank_weight.fov_down_deg", rank_fov_down_deg, -24.8);
        this->get_parameter_or<bool>("rank_weight.frame_confidence_enable", rank_frame_confidence_enable, false);
        this->get_parameter_or<double>("rank_weight.frame_alpha_rank", rank_frame_alpha_rank, 0.5);
        this->get_parameter_or<double>("rank_weight.frame_alpha_count", rank_frame_alpha_count, 0.3);
        this->get_parameter_or<double>("rank_weight.frame_alpha_residual", rank_frame_alpha_residual, 0.2);
        this->get_parameter_or<double>("rank_weight.frame_gamma", rank_frame_gamma, 2.0);
        this->get_parameter_or<double>("rank_weight.frame_ref_points", rank_frame_ref_points, 1000.0);
        this->get_parameter_or<double>("rank_weight.frame_residual_sigma", rank_frame_residual_sigma, 0.1);
        this->get_parameter_or<double>("rank_weight.frame_ref_mean", rank_frame_ref_mean, 1.5);
        this->get_parameter_or<int>("rank_weight.log_every_n_frames", rank_log_every_n_frames, 20);
        this->get_parameter_or<bool>("rank_weight.terminal_distribution_enable", rank_terminal_distribution_enable, false);
        this->get_parameter_or<int>("rank_weight.terminal_distribution_every_n_frames", rank_terminal_distribution_every_n_frames, 20);
        this->get_parameter_or<double>("rank_weight.terminal_distribution_range_bin_size", rank_terminal_distribution_range_bin_size, 10.0);
        this->get_parameter_or<double>("rank_weight.terminal_distribution_azimuth_bin_deg", rank_terminal_distribution_azimuth_bin_deg, 15.0);
        this->get_parameter_or<double>("rank_weight.terminal_distribution_max_range", rank_terminal_distribution_max_range, 0.0);
        this->get_parameter_or<bool>("adaptive_lidar_cov.enable", adaptive_lidar_cov_enable, false);
        this->get_parameter_or<string>("adaptive_lidar_cov.mode", adaptive_cov_mode, "degradation");
        this->get_parameter_or<double>("adaptive_lidar_cov.base_cov", base_lidar_cov, LASER_POINT_COV);
        this->get_parameter_or<double>("adaptive_lidar_cov.min_confidence", adaptive_cov_min_confidence, 0.1);
        this->get_parameter_or<double>("adaptive_lidar_cov.degradation_gamma", adaptive_cov_degradation_gamma, 4.0);
        this->get_parameter_or<double>("adaptive_lidar_cov.min_cov", adaptive_cov_min, LASER_POINT_COV);
        this->get_parameter_or<double>("adaptive_lidar_cov.max_cov", adaptive_cov_max, 0.1);
        this->get_parameter_or<bool>("map_confidence_fail_safe.enable", map_confidence_fail_safe_enable, true);
        this->get_parameter_or<int>("map_confidence_fail_safe.min_effective_points", map_min_effective_points, 100);
        this->get_parameter_or<int>("map_confidence_fail_safe.skip_below_effective_points", map_skip_below_effective_points, 20);
        this->get_parameter_or<double>("map_confidence_fail_safe.min_rank_valid_ratio", map_min_rank_valid_ratio, 0.2);
        this->get_parameter_or<double>("map_confidence_fail_safe.threshold_relax_factor", map_threshold_relax_factor, 0.5);
        this->get_parameter_or<bool>("experiment_log.enable", experiment_log_enable, true);
        this->get_parameter_or<string>("experiment_log.path", experiment_log_path, "");

        rank_horizontal_resolution_deg = clamp_value(rank_horizontal_resolution_deg, 0.01, 360.0);
        std::transform(rank_ring_source.begin(), rank_ring_source.end(), rank_ring_source.begin(), ::tolower);
        if (rank_ring_source == "pointcloud2_ring")
            rank_ring_source_mode = RANK_RING_POINTCLOUD2;
        else if (rank_ring_source == "elevation")
            rank_ring_source_mode = RANK_RING_ELEVATION;
        else
        {
            if (rank_ring_source != "normal_x")
                RCLCPP_WARN(this->get_logger(),
                            "Unknown rank ring_source '%s'; using normal_x",
                            rank_ring_source.c_str());
            rank_ring_source = "normal_x";
            rank_ring_source_mode = RANK_RING_NORMAL_X;
        }
        rank_ring_valid_warning_ratio = clamp_value(rank_ring_valid_warning_ratio, 0.0, 1.0);
        rank_sigma_r = std::max(rank_sigma_r, 1e-6);
        std::transform(rank_range_factor_mode.begin(), rank_range_factor_mode.end(),
                       rank_range_factor_mode.begin(), ::tolower);
        if (rank_range_factor_mode == "none")
            rank_range_factor_mode_id = 1;
        else if (rank_range_factor_mode == "inverse")
            rank_range_factor_mode_id = 2;
        else
        {
            if (rank_range_factor_mode != "linear_boost")
                RCLCPP_WARN(this->get_logger(),
                            "Unknown range_factor_mode '%s'; using linear_boost",
                            rank_range_factor_mode.c_str());
            rank_range_factor_mode = "linear_boost";
            rank_range_factor_mode_id = 0;
        }
        rank_min_weight = std::max(rank_min_weight, 1e-6);
        rank_max_weight = std::max(rank_max_weight, rank_min_weight);
        rank_huber_delta = std::max(rank_huber_delta, 1e-6);
        rank_map_filter_percent = clamp_value(rank_map_filter_percent, 0.0, 90.0);
        rank_frame_ref_points = std::max(rank_frame_ref_points, 1.0);
        rank_frame_residual_sigma = std::max(rank_frame_residual_sigma, 1e-6);
        rank_frame_ref_mean = std::max(rank_frame_ref_mean, 1e-6);
        rank_terminal_distribution_every_n_frames = std::max(rank_terminal_distribution_every_n_frames, 1);
        rank_terminal_distribution_range_bin_size = std::max(rank_terminal_distribution_range_bin_size, 0.1);
        rank_terminal_distribution_azimuth_bin_deg = clamp_value(rank_terminal_distribution_azimuth_bin_deg, 1.0, 360.0);
        rank_terminal_distribution_max_range = std::max(rank_terminal_distribution_max_range, 0.0);
        lidar_qos_depth = std::max(lidar_qos_depth, 1);
        imu_qos_depth = std::max(imu_qos_depth, 1);
        executor_threads = clamp_value(executor_threads, 2, 16);
        map_publish_period_sec = std::max(map_publish_period_sec, 0.1);
        map_visual_voxel_size = std::max(map_visual_voxel_size, 0.1);
        map_visual_max_points = std::max(map_visual_max_points, 1000);
        range_image_publish_every_n_frames = std::max(range_image_publish_every_n_frames, 1);
        range_image_row_scale = clamp_value(range_image_row_scale, 1, 32);
        range_image_max_range = std::max(range_image_max_range, 0.0);
        range_image_cloud_radius = std::max(range_image_cloud_radius, 0.1);
        range_image_cloud_height = std::max(range_image_cloud_height, 0.1);
        if (range_image_topic.empty())
            range_image_topic = "/debug/cylindrical_range_image";
        if (range_image_cloud_topic.empty())
            range_image_cloud_topic = "/debug/cylindrical_range_cloud";
        if (filtered_cloud_topic.empty())
            filtered_cloud_topic = "/cloud_weather_filtered";
        std::transform(adaptive_cov_mode.begin(), adaptive_cov_mode.end(), adaptive_cov_mode.begin(), ::tolower);
        if (adaptive_cov_mode != "confidence" && adaptive_cov_mode != "degradation")
            adaptive_cov_mode = "degradation";
        base_lidar_cov = std::max(base_lidar_cov, 1e-9);
        adaptive_cov_min_confidence = clamp_value(adaptive_cov_min_confidence, 0.01, 1.0);
        adaptive_cov_degradation_gamma = std::max(adaptive_cov_degradation_gamma, 0.0);
        adaptive_cov_min = std::max(adaptive_cov_min, 1e-9);
        adaptive_cov_max = std::max(adaptive_cov_max, adaptive_cov_min);
        adaptive_lidar_cov = base_lidar_cov;
        map_min_effective_points = std::max(map_min_effective_points, 1);
        map_skip_below_effective_points = std::max(0, std::min(map_skip_below_effective_points,
                                                               map_min_effective_points));
        map_min_rank_valid_ratio = clamp_value(map_min_rank_valid_ratio, 0.0, 1.0);
        map_threshold_relax_factor = clamp_value(map_threshold_relax_factor, 0.0, 1.0);

#ifdef FAST_LIO_USE_CUDA
        if (rank_gpu_enable)
        {
            std::string gpu_reason;
            rank_gpu_active = rank_cuda_is_available(&gpu_reason);
            if (rank_gpu_active)
                RCLCPP_INFO(this->get_logger(),
                            "[GPU] full CUDA rank pipeline enabled on %s",
                            gpu_reason.c_str());
            else
                RCLCPP_WARN(this->get_logger(), "[GPU] CUDA unavailable; using CPU rank: %s", gpu_reason.c_str());
        }
#else
        rank_gpu_active = false;
        if (rank_gpu_enable)
            RCLCPP_WARN(this->get_logger(), "[GPU] fast_lio was built without CUDA; using CPU rank");
#endif

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);
        RCLCPP_INFO(this->get_logger(),
                    "[INPUT] dataset=%s lidar=%s frame=%s imu=%s frame=%s map=%s body=%s "
                    "time_sync=%s offset=%.6f ring_source=%s",
                    dataset_name.c_str(), lid_topic.c_str(), lidar_frame_id.c_str(),
                    imu_topic.c_str(), imu_frame_id.c_str(), map_frame_id.c_str(), body_frame_id.c_str(),
                    time_sync_en ? "on" : "off",
                    time_diff_lidar_to_imu, rank_ring_source.c_str());
        RCLCPP_INFO(this->get_logger(),
                    "[ROBUST] rank=%s gpu=%s huber=%s frame_conf=%s map_filter=%s "
                    "w=[%.3f, %.3f] huber_delta=%.3f map_filter=%.1f%% init_bg_mean=%s",
                    rank_weight_enable ? "on" : "off",
                    rank_gpu_active ? "cuda" : "cpu",
                    rank_huber_enable ? "on" : "off",
                    rank_frame_confidence_enable ? "on" : "off",
                    rank_map_filter_enable ? "on" : "off",
                    rank_min_weight,
                    rank_max_weight,
                    rank_huber_delta,
                    rank_map_filter_percent,
                    init_gyro_bias_from_mean ? "on" : "off");
        RCLCPP_INFO(this->get_logger(),
                    "[SYNC] lidar_qos=%s depth=%d imu_qos=%s depth=%d",
                    lidar_qos_reliable ? "reliable" : "best_effort",
                    lidar_qos_depth,
                    imu_qos_reliable ? "reliable" : "best_effort",
                    imu_qos_depth);
        RCLCPP_INFO(this->get_logger(),
                    "[RVIZ] dense=%s body=%s map=%s period=%.1fs voxel=%.2fm max_points=%d "
                    "range_image=%s topic=%s cylinder=%s topic=%s filtered=%s topic=%s every=%d row_scale=%d "
                    "cylinder_radius=%.1f height=%.1f",
                    dense_pub_en ? "on" : "off",
                    scan_body_pub_en ? "on" : "off",
                    map_pub_en ? "on" : "off",
                    map_publish_period_sec,
                    map_visual_voxel_size,
                    map_visual_max_points,
                    range_image_pub_en ? "on" : "off",
                    range_image_topic.c_str(),
                    range_image_cloud_pub_en ? "on" : "off",
                    range_image_cloud_topic.c_str(),
                    filtered_cloud_pub_en ? "on" : "off",
                    filtered_cloud_topic.c_str(),
                    range_image_publish_every_n_frames,
                    range_image_row_scale,
                    range_image_cloud_radius,
                    range_image_cloud_height);
        RCLCPP_INFO(this->get_logger(),
                    "[ADAPTIVE_R] enable=%s mode=%s base=%.6g min=%.6g max=%.6g gamma=%.2f",
                    adaptive_lidar_cov_enable ? "on" : "off", adaptive_cov_mode.c_str(),
                    base_lidar_cov, adaptive_cov_min, adaptive_cov_max,
                    adaptive_cov_degradation_gamma);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = map_frame_id;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        if (extrinT.size() != 3 || extrinR.size() != 9)
        {
            throw std::invalid_argument(
                "mapping.extrinsic_T must contain 3 values and mapping.extrinsic_R must contain 9 values");
        }
        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
        const double extrinsic_det = Lidar_R_wrt_IMU.determinant();
        const double extrinsic_orthogonality_error =
            (Lidar_R_wrt_IMU.transpose() * Lidar_R_wrt_IMU - Eye3d).norm();
        if (!std::isfinite(extrinsic_det) || std::fabs(extrinsic_det - 1.0) > 1e-3 ||
            extrinsic_orthogonality_error > 1e-3)
        {
            throw std::invalid_argument(
                "mapping.extrinsic_R must be a proper orthonormal rotation matrix (det=+1)");
        }
        const V3D lidar_up_in_imu = Lidar_R_wrt_IMU * V3D::UnitZ();
        RCLCPP_INFO(this->get_logger(),
                    "[EXTRINSIC] T_L_I=[%.6f %.6f %.6f] det(R)=%.6f "
                    "lidar_up_in_imu=[%.6f %.6f %.6f]",
                    Lidar_T_wrt_IMU.x(), Lidar_T_wrt_IMU.y(), Lidar_T_wrt_IMU.z(),
                    extrinsic_det,
                    lidar_up_in_imu.x(), lidar_up_in_imu.y(), lidar_up_in_imu.z());
        if (lidar_up_in_imu.z() < 0.0)
        {
            RCLCPP_WARN(this->get_logger(),
                        "Configured extrinsic maps LiDAR +Z toward IMU -Z. Verify both sensor frame conventions.");
        }
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
        p_imu->set_init_gyro_bias_from_mean(init_gyro_bias_from_mean);

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        // FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        // ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        if (experiment_log_path.empty())
            experiment_log_path = root_dir + "Log/weather_adaptive_metrics.csv";
        if (experiment_log_enable)
        {
            frame_metrics_log_.open(experiment_log_path, ios::out);
            if (frame_metrics_log_)
            {
                frame_metrics_log_
                    << "timestamp,raw_point_count,downsampled_point_count,rank_valid_num,rank_valid_ratio,"
                    << "rank_mean,lidar_confidence,degradation_score,frame_rank_weight,residual_mean,"
                    << "residual_variance,adaptive_lidar_cov,map_rank_threshold,map_update_point_count,"
                    << "pose_x,pose_y,pose_z,roll,pitch,yaw,iekf_update_time,kdtree_search_time,"
                    << "kdtree_update_time,preprocess_time,rank_compute_time,lidar_gap,imu_samples,"
                    << "lidar_queue_size,imu_queue_size,lidar_received,lidar_processed,gpu_active,"
                    << "total_processing_time\n";
            }
            else
                RCLCPP_WARN(this->get_logger(), "Unable to open experiment log: %s", experiment_log_path.c_str());
        }

        /*** ROS subscribe initialization ***/
        auto lidar_qos = rclcpp::QoS(rclcpp::KeepLast(lidar_qos_depth));
        auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(imu_qos_depth));
        lidar_qos.durability_volatile();
        imu_qos.durability_volatile();
        if (lidar_qos_reliable)
            lidar_qos.reliable();
        else
            lidar_qos.best_effort();
        if (imu_qos_reliable)
            imu_qos.reliable();
        else
            imu_qos.best_effort();
        lidar_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        imu_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        processing_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions lidar_subscription_options;
        lidar_subscription_options.callback_group = lidar_callback_group_;
        rclcpp::SubscriptionOptions imu_subscription_options;
        imu_subscription_options.callback_group = imu_callback_group_;
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                lid_topic, lidar_qos, livox_pcl_cbk, lidar_subscription_options);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                lid_topic, lidar_qos, standard_pcl_cbk, lidar_subscription_options);
        }
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, imu_qos, imu_cbk, imu_subscription_options);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        if (filtered_cloud_pub_en)
            pubWeatherFilteredCloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(filtered_cloud_topic, 20);
        if (range_image_pub_en)
            pubCylindricalRangeImage_ = this->create_publisher<sensor_msgs::msg::Image>(range_image_topic, 5);
        if (range_image_cloud_pub_en)
            pubCylindricalRangeCloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(range_image_cloud_topic, 5);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        if (!lidar_frame_id.empty() && !body_frame_id.empty() &&
            lidar_frame_id != body_frame_id)
        {
            static_tf_broadcaster_ =
                std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
            geometry_msgs::msg::TransformStamped lidar_extrinsic_tf;
            lidar_extrinsic_tf.header.stamp = this->get_clock()->now();
            lidar_extrinsic_tf.header.frame_id = body_frame_id;
            lidar_extrinsic_tf.child_frame_id = lidar_frame_id;
            lidar_extrinsic_tf.transform.translation.x = Lidar_T_wrt_IMU.x();
            lidar_extrinsic_tf.transform.translation.y = Lidar_T_wrt_IMU.y();
            lidar_extrinsic_tf.transform.translation.z = Lidar_T_wrt_IMU.z();
            const Eigen::Quaterniond lidar_extrinsic_quaternion(Lidar_R_wrt_IMU);
            lidar_extrinsic_tf.transform.rotation.x = lidar_extrinsic_quaternion.x();
            lidar_extrinsic_tf.transform.rotation.y = lidar_extrinsic_quaternion.y();
            lidar_extrinsic_tf.transform.rotation.z = lidar_extrinsic_quaternion.z();
            lidar_extrinsic_tf.transform.rotation.w = lidar_extrinsic_quaternion.w();
            static_tf_broadcaster_->sendTransform(lidar_extrinsic_tf);
        }

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 1000.0));  // 1ms
        timer_ = this->create_wall_timer(
            period_ms, std::bind(&LaserMappingNode::timer_callback, this), processing_callback_group_);

        auto map_period_ms = std::chrono::milliseconds(
            static_cast<int64_t>(map_publish_period_sec * 1000.0));
        map_pub_timer_ = this->create_wall_timer(
            map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this), processing_callback_group_);

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "map_save",
            std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            processing_callback_group_);

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
        RCLCPP_INFO(this->get_logger(),
                    "[EXECUTOR] lidar, imu, and processing callback groups configured for %d threads",
                    executor_threads);
    }

    ~LaserMappingNode()
    {
#ifdef FAST_LIO_USE_CUDA
        rank_cuda_release();
#endif
        fout_out.close();
        fout_pre.close();
        fout_dbg.close();
        frame_metrics_log_.close();
        fclose(fp);
    }

    int executor_thread_count() const
    {
        return executor_threads;
    }

private:
    //*** main functions ***//
    void write_experiment_frame(double iekf_update_time, double total_processing_time)
    {
        if (!experiment_log_enable || !frame_metrics_log_)
            return;

        frame_metrics_log_ << std::setprecision(15) << Measures.lidar_end_time << ','
                           << (feats_undistort ? feats_undistort->size() : 0) << ','
                           << feats_down_size << ',' << rank_valid_num_last << ','
                           << rank_valid_ratio_last << ',' << rank_mean_last << ','
                           << lidar_confidence_last << ',' << lidar_degradation_last << ','
                           << frame_rank_weight << ',' << res_mean_last << ','
                           << residual_variance_last << ',' << adaptive_lidar_cov << ','
                           << rank_map_threshold_last << ',' << add_point_size << ','
                           << state_point.pos(0) << ',' << state_point.pos(1) << ',' << state_point.pos(2) << ','
                           << euler_cur(0) << ',' << euler_cur(1) << ',' << euler_cur(2) << ','
                           << iekf_update_time << ',' << kdtree_search_time << ','
                           << kdtree_incremental_time << ',' << preprocess_time_last << ','
                           << rank_compute_time_last << ',' << lidar_gap_last << ',' << imu_samples_last << ','
                           << lidar_queue_size_last << ',' << imu_queue_size_last << ','
                           << lidar_received_count.load() << ',' << lidar_processed_count.load() << ','
                           << (rank_gpu_active ? 1 : 0) << ',' << total_processing_time << '\n';
        frame_metrics_log_.flush();
    }

    void update_rank_terminal_distribution_params()
    {
        this->get_parameter_or<bool>("rank_weight.terminal_distribution_enable",
                                     rank_terminal_distribution_enable,
                                     rank_terminal_distribution_enable);
        this->get_parameter_or<int>("rank_weight.terminal_distribution_every_n_frames",
                                    rank_terminal_distribution_every_n_frames,
                                    rank_terminal_distribution_every_n_frames);
        this->get_parameter_or<double>("rank_weight.terminal_distribution_range_bin_size",
                                       rank_terminal_distribution_range_bin_size,
                                       rank_terminal_distribution_range_bin_size);
        this->get_parameter_or<double>("rank_weight.terminal_distribution_azimuth_bin_deg",
                                       rank_terminal_distribution_azimuth_bin_deg,
                                       rank_terminal_distribution_azimuth_bin_deg);
        this->get_parameter_or<double>("rank_weight.terminal_distribution_max_range",
                                       rank_terminal_distribution_max_range,
                                       rank_terminal_distribution_max_range);

        rank_terminal_distribution_every_n_frames = std::max(rank_terminal_distribution_every_n_frames, 1);
        rank_terminal_distribution_range_bin_size = std::max(rank_terminal_distribution_range_bin_size, 0.1);
        rank_terminal_distribution_azimuth_bin_deg = clamp_value(rank_terminal_distribution_azimuth_bin_deg, 1.0, 360.0);
        rank_terminal_distribution_max_range = std::max(rank_terminal_distribution_max_range, 0.0);
    }

    void timer_callback()
    {
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** Compute rank on deskewed raw points, then keep the best-rank point per voxel. ***/
            update_rank_terminal_distribution_params();
            rank_aware_downsample(pubCylindricalRangeImage_, pubCylindricalRangeCloud_);
            if (scan_count % 20 == 0)
            {
                RCLCPP_INFO(this->get_logger(),
                            "[POINTS] raw=%zu deskew=%zu down=%d rank_valid=%d",
                            Measures.lidar ? Measures.lidar->points.size() : 0,
                            feats_undistort ? feats_undistort->points.size() : 0,
                            feats_down_size,
                            rank_valid_num_last);
            }
            t1 = omp_get_wtime();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    if (rank_weight_enable && rank_map_filter_enable && rank_valid_num_last > 0)
                    {
                        const float map_rank_thr = rank_map_threshold();
                        PointVector init_points;
                        init_points.reserve(feats_down_size);
                        for (int i = 0; i < feats_down_size; ++i)
                        {
                            const bool rank_valid = i < static_cast<int>(point_rank_valid.size()) && point_rank_valid[i];
                            const float rank = i < static_cast<int>(point_ranks.size()) ? point_ranks[i] : 0.0f;
                            if (!rank_valid || rank < map_rank_thr)
                                continue;
                            init_points.push_back(feats_down_world->points[i]);
                        }
                        if (init_points.size() > 5)
                            ikdtree.Build(init_points);
                        else
                            ikdtree.Build(feats_down_world->points);
                    }
                    else
                    {
                        ikdtree.Build(feats_down_world->points);
                    }
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();


            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            adaptive_lidar_cov = adaptive_lidar_cov_enable ? adaptive_lidar_cov : base_lidar_cov;
            kf.update_iterated_dyn_share_modified(adaptive_lidar_cov, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();

            if (lidar_processed_count.load() % 100 == 0)
            {
                RCLCPP_INFO(this->get_logger(),
                            "[SCAN_COUNT] received=%lu processed=%lu lidar_queue=%zu "
                            "imu_received=%lu imu_queue=%zu lidar_gap=%.6f",
                            static_cast<unsigned long>(lidar_received_count.load()),
                            static_cast<unsigned long>(lidar_processed_count.load()),
                            lidar_queue_size_last,
                            static_cast<unsigned long>(imu_received_count.load()),
                            imu_queue_size_last,
                            lidar_gap_last);
            }

            if (rank_weight_enable && rank_log_every_n_frames > 0 && scan_count % rank_log_every_n_frames == 0)
            {
                RCLCPP_INFO(this->get_logger(),
                            "[RANK] raw=%zu down=%d valid=%d ratio=%.3f eff=%d mean=%.3f "
                            "c_k=%.3f D_k=%.3f R=%.6g map_thr=%.3f frame_w=%.3f add=%d",
                            feats_undistort->points.size(),
                            feats_down_size,
                            rank_valid_num_last,
                            rank_valid_ratio_last,
                            effct_feat_num,
                            rank_mean_last,
                            lidar_confidence_last,
                            lidar_degradation_last,
                            adaptive_lidar_cov,
                            rank_map_threshold_last,
                            frame_rank_weight,
                            add_point_size);
            }

            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath_);
            publish_weather_filtered_world(pubWeatherFilteredCloud_);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
            write_experiment_frame(t_update_end - t_update_start, omp_get_wtime() - t0);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        publish_map(pubLaserCloudMap_);
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (map_file_path.empty())
        {
            res->success = false;
            res->message = "Map file path is empty.";
            return;
        }
        if (pcl_wait_pub->empty())
        {
            res->success = false;
            res->message = "Map is empty. Wait until /Laser_map has accumulated points.";
            return;
        }

        save_to_pcd();
        res->success = true;
        res->message = "Map saved to " + map_file_path;
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubWeatherFilteredCloud_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pubCylindricalRangeImage_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCylindricalRangeCloud_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;
    rclcpp::CallbackGroup::SharedPtr lidar_callback_group_;
    rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
    rclcpp::CallbackGroup::SharedPtr processing_callback_group_;

    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;
    ofstream frame_metrics_log_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    signal(SIGINT, SigHandle);

    auto node = std::make_shared<LaserMappingNode>();
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), node->executor_thread_count());
    executor.add_node(node);
    executor.spin();
    executor.remove_node(node);

    if (rclcpp::ok())
        rclcpp::shutdown();

    /**************** save trajectory ****************/
    if(traj_save_en){
        save_trajectory(traj_file_path);
        std::cout << "Save FAST-LIO2 trajectory !!" << std::endl;
    }

    /**************** save map ****************/
    /* Make sure you have enough memories to save the map */
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to " << map_file_path <<endl;
        pcd_writer.writeBinary(map_file_path, *pcl_wait_save);
    }

    /**************** save runtime log ****************/
    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
