#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
exec docker build --network host -t karmishthannauth/minddrive_env_ros:v01 .
