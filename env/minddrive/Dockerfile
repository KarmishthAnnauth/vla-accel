FROM karmishthannauth/orion_env_ros:v01

RUN pip install --no-cache-dir "transformers==4.45.2" \
 && pip install --no-cache-dir --no-deps \
        stable_baselines3 gymnasium farama-notifications cloudpickle \
        lyft_dataset_sdk fire cachetools \
        pyzmq black mypy_extensions pathspec pytokens \
 && python3 -c "import torch, transformers, tokenizers; assert torch.__version__.startswith('2.4.0a0'), torch.__version__; print('torch', torch.__version__, 'transformers', transformers.__version__)"

RUN sed -i 's#^/root/Orion$#/benchmarking/MindDrive#' /usr/local/lib/python3.10/dist-packages/easy-install.pth \
 && printf '/benchmarking/MindDrive\n.' > /usr/local/lib/python3.10/dist-packages/mmcv.egg-link \
 && cat /usr/local/lib/python3.10/dist-packages/easy-install.pth

ENV VLA_ACCEL_MODEL=minddrive
