import os
import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')
    workspace_path = os.environ.get(
        'FAST_LIO_WS',
        '/home/yoon/FAST_LIO_ROS2/ros2_ws_from_container'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz_use = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    rviz_software_rendering = LaunchConfiguration('rviz_software_rendering')
    play_bag = LaunchConfiguration('play_bag')
    bag_path = LaunchConfiguration('bag_path')
    bag_rate = LaunchConfiguration('bag_rate')
    config_path = LaunchConfiguration('config_path')
    qos_profile_path = LaunchConfiguration('qos_profile_path')
    lidar_topic = LaunchConfiguration('lidar_topic')
    imu_topic = LaunchConfiguration('imu_topic')
    lidar_frame_id = LaunchConfiguration('lidar_frame_id')
    imu_frame_id = LaunchConfiguration('imu_frame_id')
    time_sync_enable = LaunchConfiguration('time_sync_enable')
    lidar_imu_time_offset = LaunchConfiguration('lidar_imu_time_offset')
    lidar_qos_depth = LaunchConfiguration('lidar_qos_depth')
    imu_qos_depth = LaunchConfiguration('imu_qos_depth')
    rank_weight_enable = LaunchConfiguration('rank_weight_enable')
    rank_gpu_enable = LaunchConfiguration('rank_gpu_enable')
    rank_ring_source = LaunchConfiguration('rank_ring_source')
    rank_huber_enable = LaunchConfiguration('rank_huber_enable')
    rank_frame_confidence_enable = LaunchConfiguration('rank_frame_confidence_enable')
    rank_map_filter_enable = LaunchConfiguration('rank_map_filter_enable')
    init_gyro_bias_from_mean = LaunchConfiguration('init_gyro_bias_from_mean')
    adaptive_lidar_cov_enable = LaunchConfiguration('adaptive_lidar_cov_enable')

    default_sequence_name = 'boreas-2021-07-20-17-33'
    default_config_path = os.path.join(package_path, 'config', 'boreas_2021_07_20_17_33.yaml')
    default_rviz_config_path = os.path.join(package_path, 'rviz_cfg', 'fastlio.rviz')
    default_bag_path = os.path.join(workspace_path, 'datasets', default_sequence_name)
    default_qos_profile_path = os.path.join(workspace_path, 'config', 'boreas_rosbag_qos.yaml')

    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[
            config_path,
            {
                'use_sim_time': use_sim_time,
                'rank_weight.enable': rank_weight_enable,
                'rank_weight.gpu_enable': rank_gpu_enable,
                'rank_weight.ring_source': rank_ring_source,
                'rank_weight.huber_enable': rank_huber_enable,
                'rank_weight.frame_confidence_enable': rank_frame_confidence_enable,
                'rank_weight.map_filter_enable': rank_map_filter_enable,
                'mapping.init_gyro_bias_from_mean': init_gyro_bias_from_mean,
                'adaptive_lidar_cov.enable': adaptive_lidar_cov_enable,
                'common.lid_topic': lidar_topic,
                'common.imu_topic': imu_topic,
                'common.lid_frame_id': lidar_frame_id,
                'common.imu_frame_id': imu_frame_id,
                'common.time_sync_en': time_sync_enable,
                'common.time_offset_lidar_to_imu': lidar_imu_time_offset,
                'common.lidar_qos_depth': lidar_qos_depth,
                'common.imu_qos_depth': imu_qos_depth,
            },
        ],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz_use)
    )

    bag_player = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play', bag_path,
                    '--clock',
                    '--rate', bag_rate,
                    '--topics', lidar_topic, imu_topic,
                    '--qos-profile-overrides-path', qos_profile_path,
                ],
                output='screen',
                condition=IfCondition(play_bag)
            )
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('config_path', default_value=default_config_path),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_cfg', default_value=default_rviz_config_path),
        DeclareLaunchArgument('rviz_software_rendering', default_value='false'),
        DeclareLaunchArgument('play_bag', default_value='true'),
        DeclareLaunchArgument('bag_path', default_value=default_bag_path),
        DeclareLaunchArgument('bag_rate', default_value='0.5'),
        DeclareLaunchArgument('qos_profile_path', default_value=default_qos_profile_path),
        DeclareLaunchArgument('lidar_topic', default_value='/velodyne_points'),
        DeclareLaunchArgument('imu_topic', default_value='/handsfree/imu'),
        DeclareLaunchArgument('lidar_frame_id', default_value='lidar'),
        DeclareLaunchArgument('imu_frame_id', default_value='imu'),
        DeclareLaunchArgument('time_sync_enable', default_value='false'),
        DeclareLaunchArgument('lidar_imu_time_offset', default_value='0.0'),
        DeclareLaunchArgument('lidar_qos_depth', default_value='50'),
        DeclareLaunchArgument('imu_qos_depth', default_value='5000'),
        DeclareLaunchArgument('rank_weight_enable', default_value='false'),
        DeclareLaunchArgument('rank_gpu_enable', default_value='true'),
        DeclareLaunchArgument('rank_ring_source', default_value='pointcloud2_ring'),
        DeclareLaunchArgument('rank_huber_enable', default_value='false'),
        DeclareLaunchArgument('rank_frame_confidence_enable', default_value='false'),
        DeclareLaunchArgument('rank_map_filter_enable', default_value='true'),
        DeclareLaunchArgument('init_gyro_bias_from_mean', default_value='false'),
        DeclareLaunchArgument('adaptive_lidar_cov_enable', default_value='false'),
        SetEnvironmentVariable(
            name='LIBGL_ALWAYS_SOFTWARE',
            value='1',
            condition=IfCondition(rviz_software_rendering)
        ),
        SetEnvironmentVariable(
            name='QT_X11_NO_MITSHM',
            value='1',
            condition=IfCondition(rviz_software_rendering)
        ),
        SetEnvironmentVariable(
            name='__GLX_VENDOR_LIBRARY_NAME',
            value='nvidia',
            condition=UnlessCondition(rviz_software_rendering)
        ),
        SetEnvironmentVariable(
            name='__NV_PRIME_RENDER_OFFLOAD',
            value='1',
            condition=UnlessCondition(rviz_software_rendering)
        ),
        fast_lio_node,
        rviz_node,
        bag_player,
    ])
