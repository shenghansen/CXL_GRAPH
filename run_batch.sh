cd ~/Gemini/build
cmake -DGLOBAL_STEALING_DENSE ..
ninja
cd ..
./run.script >> output.log.22 2>&1 


cd ~/Gemini/build
cmake -DSPARSE_MODE_UNIDIRECTIONAL -DDENSE_MODE_UNIDIRECTIONAL ..
ninja
cd ..
./run.script >> output.log.batch.1 2>&1 