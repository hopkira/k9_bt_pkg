from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_trees_ros_viewer',
            executable='py-trees-tree-viewer',
            name='pytrees_viewer',
            output='screen'
        )
    ])