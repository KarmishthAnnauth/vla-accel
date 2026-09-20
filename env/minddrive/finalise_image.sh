#!/usr/bin/env bash
set -e
SP=/usr/local/lib/python3.10/dist-packages
sed -i 's#^/root/Orion$#/benchmarking/MindDrive#' $SP/easy-install.pth
printf '/benchmarking/MindDrive\n.' > $SP/mmcv.egg-link
grep -q '^/benchmarking/MindDrive$' $SP/easy-install.pth
rm -rf /root/.cache/pip
cd /
python3 - <<'PY'
import mmcv, torch, transformers, mmcv._ext, peft
assert mmcv.__file__.startswith("/benchmarking/MindDrive/"), mmcv.__file__
assert torch.__version__.startswith("2.4.0a0"), torch.__version__
assert transformers.__version__ == "4.45.2", transformers.__version__
print("finalise: mmcv", mmcv.__file__, "| torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
PY
