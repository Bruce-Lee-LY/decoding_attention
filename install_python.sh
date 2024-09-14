# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 21:14:13 on Tue, Oct 31, 2023
#
# Description: insatll python script

#!/bin/bash

set -euo pipefail

echo "========== intsall enter =========="

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

echo_cmd() {
    echo $1
    $1
}

echo "========== intsall decoding_attention =========="

echo_cmd "rm -rf build dist decoding_attn.egg-info"
echo_cmd "python3 setup.py install --user --prefix="

echo "========== intsall exit =========="
