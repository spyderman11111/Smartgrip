# gripanything/launch/camera_tf.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 可改的 Launch 参数（默认值用你给的手眼标定）
    parent = LaunchConfiguration('parent')
    child  = LaunchConfiguration('child')
    x  = LaunchConfiguration('x')
    y  = LaunchConfiguration('y')
    z  = LaunchConfiguration('z')
    qx = LaunchConfiguration('qx')
    qy = LaunchConfiguration('qy')
    qz = LaunchConfiguration('qz')
    qw = LaunchConfiguration('qw')

    return LaunchDescription([
        DeclareLaunchArgument('parent', default_value='tool0',
                              description='Parent frame id'),
        DeclareLaunchArgument('child',  default_value='camera_link',
                              description='Child frame id'),

        # 下面 8 个默认值就是你提供的 hand_eye_calibration
        DeclareLaunchArgument('x',  default_value='-0.000006852374024'),
        DeclareLaunchArgument('y',  default_value='-0.099182661943126947'),
        DeclareLaunchArgument('z',  default_value='0.02391824813032688'),
        DeclareLaunchArgument('qx', default_value='-0.0036165657530785695'),
        DeclareLaunchArgument('qy', default_value='-0.000780788838366878'),
        DeclareLaunchArgument('qz', default_value='0.7078681983794892'),
        DeclareLaunchArgument('qw', default_value='0.7063348529868249'),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tool0_to_camera_link_tf',
            # static_transform_publisher 参数顺序：
            # x y z qx qy qz qw parent_frame child_frame
            arguments=[x, y, z, qx, qy, qz, qw, parent, child],
            output='screen'
        )
    ])
