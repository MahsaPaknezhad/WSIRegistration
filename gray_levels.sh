#!/bin/sh
if [ -n "$LD_LIBRARY_PATH" ]; then
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/opencv/build-3.4.1/lib
else
  LD_LIBRARY_PATH=/home/opencv/build-3.4.1/lib
fi
export LD_LIBRARY_PATH
exec /home/images/gray_levels "$@ $1 $2 $3 $4 $5 $6 $7 $8"
