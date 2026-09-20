import os
from glob import glob

from setuptools import find_packages, setup

package_name = "orion_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="orion_ros maintainer",
    maintainer_email="user@example.com",
    description="ROS 2 interface for ORION VLA inference.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "orion_node = orion_ros.orion_node:main",
            "orion_withpid_node = orion_ros.orion_withpid_node:main",
            "orion_lite_node = orion_ros.orion_lite_rosnode:main",
            "carla_trajectory_viz_node = orion_ros.carla_trajectory_viz_node:main",
            "image_decompress_node = orion_ros.image_decompress_node:main",
            "stanley_controller_node = orion_ros.stanley_controller_node:main",
        ],
    },
)
