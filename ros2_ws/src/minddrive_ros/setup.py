import os
from glob import glob

from setuptools import find_packages, setup

package_name = "minddrive_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "tools"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="minddrive_ros maintainer",
    maintainer_email="user@example.com",
    description="ROS 2 interface for MindDrive VLA inference.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "minddrive_node = minddrive_ros.minddrive_node:main",
            "image_decompress_node = minddrive_ros.image_decompress_node:main",
        ],
    },
)
