cd ~/Work/ggmlpython/build
make clean
cmake ..
make
cd ../examples/stablelm
python GGMLWrapper.py