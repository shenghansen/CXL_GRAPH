#!/bin/bash

# Commands and arguments for testing
cd ~/Gemini/build
cmake -DSPARSE_MODE_UNIDIRECTIONAL=ON ..
ninja
cd ..
echo "123456" | sudo ./run.script > hl_sparse_mode.log 2>&1

cd ~/Gemini/build
cmake -DDENSE_MODE_UNIDIRECTIONAL=ON ..
ninja
cd ..
echo "123456" | sudo ./run.script > hl_dense_mode.log 2>&1

cd ~/Gemini/build
cmake -DSPARSE_MODE_UNIDIRECTIONAL=ON -DDENSE_MODE_UNIDIRECTIONAL=ON ..
ninja
cd ..
echo "123456" | sudo ./run.script > hl_sparse_dense.log 2>&1

cd ~/Gemini/build
cmake -DSPARSE_MODE_UNIDIRECTIONAL=ON -DGET=ON ..
ninja
cd ..
echo "123456" | sudo ./run.script > hl_sparse_mode_get.log 2>&1

cd ~/Gemini/build
cmake -DSPARSE_MODE_UNIDIRECTIONAL=ON -DDENSE_MODE_UNIDIRECTIONAL=ON -DGET=ON ..
ninja
cd ..
echo "123456" | sudo ./run.script > hl_final_get.log 2>&1