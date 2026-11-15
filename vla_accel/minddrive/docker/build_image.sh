#!/usr/bin/env bash
# Build the MindDrive bench image (arm64, on the Orin).
set -e
cd "$(dirname "$0")"
exec docker build --network host -t "${IMAGE:-karmishthannauth/minddrive_env:v01}" .
