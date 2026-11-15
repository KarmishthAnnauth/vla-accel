# MindDrive bench image: the ORION image (torch 2.4.0a0 nv24.07, mmcv build) + transformers 4.45.2.
ARG BASE=karmishthannauth/orion_env_ros:v01
FROM ${BASE}
ARG MINDDRIVE_CTR=/workspace/MindDrive

RUN pip install --no-cache-dir "transformers==4.45.2" \
 && pip install --no-cache-dir --no-deps \
        stable_baselines3 gymnasium farama-notifications cloudpickle \
        lyft_dataset_sdk fire cachetools \
        pyzmq black mypy_extensions pathspec pytokens \
 && python3 -c "import torch, transformers, tokenizers; assert torch.__version__.startswith('2.4.0a0'), torch.__version__; print('torch', torch.__version__, 'transformers', transformers.__version__)"

# Point the base image's develop-mode mmcv at the MindDrive checkout mounted at MINDDRIVE_CTR.
RUN sed -i "s#^/root/Orion\$#${MINDDRIVE_CTR}#" /usr/local/lib/python3.10/dist-packages/easy-install.pth \
 && printf "${MINDDRIVE_CTR}\n." > /usr/local/lib/python3.10/dist-packages/mmcv.egg-link \
 && grep -qx "${MINDDRIVE_CTR}" /usr/local/lib/python3.10/dist-packages/easy-install.pth \
 && rm -rf /root/.cache/pip

ENV VLA_ACCEL_MODEL=minddrive
