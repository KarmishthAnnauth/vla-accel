import os
from glob import glob

from setuptools import find_packages, setup

package_name = "simlingo_ros"

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
    maintainer="user",
    maintainer_email="user@example.com",
    description="ROS 2 interface for SimLingo VLA inference.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "simlingo_node = simlingo_ros.simlingo_node:main",
            "stanley_controller_node = simlingo_ros.stanley_controller_node:main",
        ],
    },
)
