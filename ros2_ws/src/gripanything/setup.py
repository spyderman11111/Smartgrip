from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'gripanything'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MA_SmartGrip',
    maintainer_email='shuo.zhang@campus.tu-berlin.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'seeanything = gripanything.seeanything:main',
            'seeanything_debug = gripanything.seeanything_debug:main',
            'tool_to_camera_tf_publisher = gripanything.utils.tool_to_camera_tf_publisher:main',
            'goto_point_from_object_json = gripanything.utils.goto_point_from_object_json:main',
            'publish_object_points_tf = gripanything.utils.publish_object_points_tf:main',
        ],
    },
)
