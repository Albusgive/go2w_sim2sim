# MUJOCO
## CPP
### 依赖
#### MUJOCO 编译安装
* 克隆项目
  ``` git clone https://github.com/google-deepmind/mujoco.git ```
* 编译（建议开启魔法）
```
cd mujoco
mkdir build
cd build
cmake ..
cmake --build . 多线程编译使用 cmake --build . -j线程数
```  
* 选择安装位置（推荐/opt）  
  `cmake -DCMAKE_INSTALL_PREFIX=/opt/mujoco .`
* `sudo cmake --install .`

#### MUJOCO Release版本
替换CMakeLists.txt中寻找mujoco部分
```CMake
set(MUJOCO_PATH "your mujoco path")
include_directories(${MUJOCO_PATH}/include)
link_directories(${MUJOCO_PATH}/build/bin)
set(MUJOCO_LIB ${MUJOCO_PATH}/build/lib/libmujoco.so)
```
链接库部分中 ${MUJOCO_LIB}
`target_link_libraries(your_app ${MUJOCO_LIB} glut GL GLU glfw)`
#### Libtorch
**下载**
`wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip`
**解压后在~/.bashrc中配置路径**
`export Torch_DIR=/your_path/libtorch`
`source ~/.bashrc`

#### GamePad支持
```
sudo apt-get install parcellite
sudo apt-get install libudev-dev
sudo apt-get install joystick
```
### 编译
```
cd mujoco/C++
mkdir build
cd build
cmake ..
make
```
