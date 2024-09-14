# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 21:08:30 on Sun, Aug 27, 2023
#
# Description: format script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

find . \( -name '*.c' -or -name '*.cpp' -or -name '*.cc' -or -name '*.cxx' -or -name '*.cu' -or -name '*.h' -or -name '*.hpp' -or -name '*.cuh' -or -name '*.inl' \) -exec clang-format -style=file -i {} \;
