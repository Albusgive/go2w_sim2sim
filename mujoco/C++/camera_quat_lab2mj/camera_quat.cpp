#include <iostream>
#include <iomanip>
#include <cmath>
#include <mujoco/mujoco.h>

int main() {
    // 1. Isaac Lab 中的相机姿态 (w, x, y, z)
    // 根据输入: rot = (cos(12.5), 0, sin(12.5), 0)
    double q_isaac[4] = {std::cos(12.5 * M_PI / 180.0), 0.0, std::sin(12.5 * M_PI / 180.0), 0.0};

    // 2. 构造 Offset 四元数
    // 针对“竖着向右看”的反转逻辑：
    // 我们需要把 MuJoCo 的 Forward (-z) 映射到 Isaac 的 Forward (+x)
    // 我们需要把 MuJoCo 的 Up (+y) 映射到 Isaac 的 Up (+z)
    // 我们需要把 MuJoCo 的 Right (+x) 映射到 Isaac 的 Right (-y)
    
    // 重新排列矩阵 (列向量形式)：
    // Col0: MuJoCo 的 X 轴在 Isaac 空间的方向
    // Col1: MuJoCo 的 Y 轴在 Isaac 空间的方向
    // Col2: MuJoCo 的 Z 轴在 Isaac 空间的方向
    double R_offset[9] = {
         0,  0, -1,  // Isaac 的 X 对应 MuJoCo 的 -Z
        -1,  0,  0,  // Isaac 的 Y 对应 MuJoCo 的 -X
         0,  1,  0   // Isaac 的 Z 对应 MuJoCo 的 +Y
    };

    double q_offset[4];
    mju_mat2Quat(q_offset, R_offset);

    // 3. 计算最终姿态
    // 注意：如果是从 Isaac 导出的全局姿态，通常需要 q_isaac * q_offset
    double q_final[4];
    mju_mulQuat(q_final, q_isaac, q_offset);

    // 4. 如果还是不对，尝试交换顺序 (因为有些系统是左乘)
    // mju_mulQuat(q_final, q_offset, q_isaac);

    std::cout << "MuJoCo Quat (w,x,y,z): " << std::endl;
    for(int i=0; i<4; ++i) 
        std::cout << std::fixed << std::setprecision(6) << q_final[i] << (i==3 ? "" : ", ");
    std::cout << std::endl;

    return 0;
}