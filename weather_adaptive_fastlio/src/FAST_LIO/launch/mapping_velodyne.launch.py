import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition

from launch_ros.actions import Node, SetUseSimTime


def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')
    workspace_path = os.environ.get(
        'FAST_LIO_WS',
        os.path.join(os.path.expanduser('~'), 'ros2_ws')
    )

    # Create LaunchConfigurations for the parameters with default values (not in config file)
    feature_extract_enable_param = LaunchConfiguration('feature_extract_enable', default='false') # bool
    point_filter_num_param = LaunchConfiguration('point_filter_num', default='4')                 # int
    max_iteration_param = LaunchConfiguration('max_iteration', default='3')                       # int
    filter_size_surf_param = LaunchConfiguration('filter_size_surf', default='0.5')               # double
    filter_size_map_param = LaunchConfiguration('filter_size_map', default='0.5')                 # double
    cube_side_length_param = LaunchConfiguration('cube_side_length', default='1000.0')            # double
    runtime_pos_log_enable_param = LaunchConfiguration('runtime_pos_log_enable', default='false')  # bool
    path_en_param = LaunchConfiguration('path_en', default='true')                                # bool
    scan_publish_en_param = LaunchConfiguration('scan_publish_en', default='true')                # bool
    dense_publish_en_param = LaunchConfiguration('dense_publish_en', default='true')              # bool
    scan_bodyframe_pub_en_param = LaunchConfiguration('scan_bodyframe_pub_en', default='true')    # bool
    rank_weight_enable_param = LaunchConfiguration('rank_weight_enable', default='false')         # bool
    map_file_path_param = LaunchConfiguration(
        'map_file_path',
        default=os.path.join(workspace_path, 'src', 'FAST_LIO', 'PCD', 'result.pcd')
    )

    default_config_path = os.path.join(package_path, 'config', 'velodyne.yaml')
    default_rviz_config_path = os.path.join(
        package_path, 'rviz_cfg', 'fastlio.rviz')


    use_sim_time = LaunchConfiguration('use_sim_time')
    config_path = LaunchConfiguration('config_path')
    rviz_use = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    rviz_software_rendering = LaunchConfiguration('rviz_software_rendering')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    )
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Use RViz to monitor results'
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value=default_rviz_config_path,
        description='RViz config file path'
    )
    declare_rviz_software_rendering_cmd = DeclareLaunchArgument(
        'rviz_software_rendering', default_value='false',
        description='Use Mesa software OpenGL for RViz when GPU acceleration is unavailable'
    )

    rviz_software_gl_cmd = SetEnvironmentVariable(
        name='LIBGL_ALWAYS_SOFTWARE',
        value='1',
        condition=IfCondition(rviz_software_rendering)
    )
    rviz_no_mitshm_cmd = SetEnvironmentVariable(
        name='QT_X11_NO_MITSHM',
        value='1',
        condition=IfCondition(rviz_software_rendering)
    )
    rviz_nvidia_glx_cmd = SetEnvironmentVariable(
        name='__GLX_VENDOR_LIBRARY_NAME',
        value='nvidia',
        condition=UnlessCondition(rviz_software_rendering)
    )
    rviz_nvidia_prime_cmd = SetEnvironmentVariable(
        name='__NV_PRIME_RENDER_OFFLOAD',
        value='1',
        condition=UnlessCondition(rviz_software_rendering)
    )

    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[config_path,
                    {'use_sim_time': use_sim_time,
                    'feature_extract_enable': feature_extract_enable_param,
                    'point_filter_num': point_filter_num_param,
                    'max_iteration': max_iteration_param,
                    'filter_size_surf': filter_size_surf_param,
                    'filter_size_map': filter_size_map_param,
                    'cube_side_length': cube_side_length_param,
                    'runtime_pos_log_enable': runtime_pos_log_enable_param,
                    'publish.path_en': path_en_param,
                    'publish.scan_publish_en': scan_publish_en_param,
                    'publish.dense_publish_en': dense_publish_en_param,
                    'publish.scan_bodyframe_pub_en': scan_bodyframe_pub_en_param,
                    'rank_weight.enable': rank_weight_enable_param,
                    'map_file_path': map_file_path_param}],
        output='screen'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz_use)
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)
    ld.add_action(declare_rviz_software_rendering_cmd)
    ld.add_action(rviz_software_gl_cmd)
    ld.add_action(rviz_no_mitshm_cmd)
    ld.add_action(rviz_nvidia_glx_cmd)
    ld.add_action(rviz_nvidia_prime_cmd)

    ld.add_action(fast_lio_node)
    ld.add_action(rviz_node)

    return ld
