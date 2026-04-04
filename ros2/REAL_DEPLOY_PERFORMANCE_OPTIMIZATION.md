# ROS2 真机部署性能优化分析

本文档针对当前 `ros2/src/src/go2w_real_deploy` 这套真机部署代码，整理可行的性能优化点。

目标不是立即改代码，而是先把：

- 当前性能开销主要来自哪里
- 每个优化点为什么值得做
- 建议怎么改
- 风险和副作用是什么
- 推荐的落地顺序

一次性说清楚，方便后续逐项推进。

## 1. 分析范围

本次分析重点覆盖以下路径：

- `ros2/src/src/go2w_real_deploy/go2w_real_deploy.cpp`
- `ros2/src/src/go2w_real_deploy/go2w_real_deploy_node.cpp`
- `ros2/src/include/go2w_real_deploy/go2w_real_deploy_node.h`
- `utils/cpp_manager_env/ManagerEnv.cpp`
- `utils/cpp_manager_env/ManagerEnv.hpp`
- `utils/cpp_manager_env/Buffer.hpp`
- `utils/cpp_manager_env/SimpleTensor.hpp`
- `utils/cpp_manager_env/net.cpp`

重点关注的运行链路是：

1. ROS2 回调接收 `/lowstate`、`/wirelesscontroller`、图像话题
2. `20 ms` 控制定时器触发 `low_cmd_write()`
3. 更新视觉观测
4. `manager_step(policy_id_)`
5. ONNX 推理
6. 组装 `low_cmd`
7. 发送 `/lowcmd`

## 2. 当前总体判断

当前实现已经具备可用性和一定的稳定性，但从性能角度看，仍然存在比较明确的优化空间。

总体上可以把优化机会分成四类：

1. 不必要的观测更新
2. 图像链路中的重复计算和重复拷贝
3. ROS2 执行模型没有真正发挥多线程优势
4. `ManagerEnv` / `SimpleTensor` / ONNX Runtime 这一层存在较多临时分配

如果只追求“最少改动就拿到明显收益”，优先级最高的是：

1. 只更新当前 policy 的 obs
2. 减少视觉观测在每个控制 tick 的重复刷新
3. 把深度图归一化结果缓存到“新帧到来时”而不是“控制循环里每次重算”
4. 拆 callback group，让当前的 `MultiThreadedExecutor(2)` 真正并行起来

## 3. 优先级最高的优化点

### 3.1 `manager_step()` 默认更新所有 policy 的 obs

#### 当前行为

`ManagerBasedEnv::manager_step(int id)` 里当前逻辑是：

- 先执行 `apply_policy_runtime_controls(id)`
- 再更新所有 policy 的 obs
- 最后只对当前 `id` 对应的 policy 做推理

对应位置：

- `utils/cpp_manager_env/ManagerEnv.cpp`
- `ManagerBasedEnv::manager_step`

而且 `ManagerBasedEnv` 中 `update_all_policy_obs_in_manager_step_` 默认值是 `true`，当前真机部署入口没有显式把它关掉。

#### 为什么这是大头

当前部署同时注册了 4 个 policy：

- `motion_mlp`
- `vtm`
- `vtm_lstm_sru`
- `vtm_gru_sru`

这意味着每个控制周期都会：

- 更新 `motion_mlp` 的 obs
- 更新 `vtm` 的 obs
- 更新 `vtm_lstm_sru` 的 obs
- 更新 `vtm_gru_sru` 的 obs

即使当前只激活了其中一个 policy，其余 3 个 policy 仍然会参与 obs 更新。

尤其视觉 policy 的 obs 更新比纯 proprio policy 更重，因为它们会涉及图像 term、history buffer、flatten/cat 等操作。

#### 优化思路

把 `update_all_policy_obs_in_manager_step_` 改为仅更新当前 policy：

- 正常运行时：只 `computeObs(active_policy_id)`
- 切换 policy 时：再做一次 `warm_start_history`
- 切换 policy 时：必要时配合 `reset_observation_buffers(requested_policy_id)`

也就是说，把“平时所有 policy 都维护热 obs”改成“只维护当前 policy，切换时补一次状态恢复”。

#### 预期收益

- 直接减少每 tick 的 obs 计算量
- 对视觉 policy 尤其有效
- 在 Jetson NX / Orin NX 这类边缘设备上收益通常比较明显

#### 风险与副作用

切换 policy 的瞬间，新的 policy 不再拥有一直被后台维护的 history。

不过当前代码已经有这些辅助机制：

- `refresh_visual_observations(warm_start_history)`
- `reset_observation_buffers(...)`
- `on_policy_runtime_state_reset(...)`

所以这类风险是可控的。

#### 推荐程度

非常推荐，应该优先处理。

---

### 3.2 控制循环一开始就刷新视觉观测，即使当前不需要视觉

#### 当前行为

在 `Go2wRealDeployNode::low_cmd_write()` 里，一上来就执行：

- `apply_pending_runtime_changes()`
- `refresh_visual_observations(false)`

然后才进入：

- `has_low_state_` 判定
- `is_stop_` 判定
- `manager_step(policy_id_)`

也就是说，即使当前：

- 机器人还没拿到低层状态
- 机器人处于 stop 状态
- 当前 policy 是 `motion_mlp`

仍然会先刷新视觉观测。

#### 为什么有浪费

视觉观测刷新包含：

- 调用每个 `ImageObservationTerm::compute_obs()`
- 更新 current frame
- 维护 history
- 可能后续还要被 `computeObs()` 再次读取、拼接

如果当前 active policy 根本不是视觉 policy，那么这些操作对输出控制命令没有直接贡献。

#### 优化思路

把视觉观测刷新拆成更细的条件：

1. 当前 active policy 是视觉 policy 时才正常刷新
2. 当前 active policy 不是视觉 policy 时不刷新视觉 obs
3. 只有在以下情况才强制刷新：
   - 切到视觉 policy
   - 重置视觉 policy
   - 手动打开 sensor 后需要 warm start
   - 初始化时当前就是视觉 policy

#### 预期收益

- 纯 `motion_mlp` 运行时能显著减少无用图像计算
- stop 状态下也能减少 CPU 消耗

#### 风险与副作用

如果你仍然希望“视觉 policy 随时一切换就带着最新 history 进入运行”，就需要在 policy 切换时补充 warm start。

当前代码结构已经支持这一点，因此风险低。

#### 推荐程度

非常推荐，和上一项可以一起做。

---

### 3.3 深度图归一化和复制发生在控制循环，而不是新帧到来时

#### 当前行为

当前深度图链路大致是：

1. `depth_callback()` 中把原始图处理成：
   - 毫米转米
   - resize + crop 到 `32x18`
2. 保存为 `latest_depth_image_m_`
3. 在 `build_normalized_depth_image()` 里：
   - clone 一份 `cv::Mat`
   - 检查 stale
   - 对每个像素做 clamp
   - 对每个像素做 `[min_dist, max_dist] -> [0,1]`
   - 构造新的 `std::vector<float>`
   - 最终再封装成 `SimpleTensor`

#### 为什么有浪费

如果相机是 `30 Hz` 或 `60 Hz`，而控制循环是 `50 Hz`，很多控制周期其实在重复消费同一帧图像。

也就是说当前实现会在“同一帧没有变化”的情况下，反复：

- clone 图像
- 重新归一化
- 重新分配 `std::vector<float>`
- 重新构造 `SimpleTensor`

#### 优化思路

把图像处理分成两层缓存：

1. 原始缓存：
   - `latest_depth_image_m_`
2. 已归一化缓存：
   - `latest_depth_obs_tensor_`
   - 或者 `latest_depth_obs_vector_`

然后只在新深度帧到来时更新归一化缓存。

控制循环中：

- 只判断这帧是否 stale
- 如果不 stale，直接返回缓存好的归一化 tensor
- 如果 stale，直接返回全 0 tensor

进一步可以增加一个 `depth_frame_seq_` 或 `last_processed_depth_stamp_`，明确区分“新帧”和“旧帧复用”。

#### 预期收益

- 降低控制线程上的图像 CPU 消耗
- 降低短周期重复内存分配
- 对视觉 policy 尤其有效

#### 风险与副作用

需要注意线程安全，避免：

- 图像回调正在更新缓存
- 控制线程同时读取缓存

这可以通过双缓冲或小范围加锁解决。

#### 推荐程度

非常推荐，是视觉链路里最值得做的一项。

---

### 3.4 `MultiThreadedExecutor(2)` 没有真正发挥作用

#### 当前行为

入口使用了：

- `rclcpp::executors::MultiThreadedExecutor(..., 2)`

但当前节点里没有显式定义 callback groups。

#### 为什么这意味着优化空间

在 ROS2 中，单纯把 executor 设成多线程，并不等于所有回调都会高效并行。

如果回调都落在默认 group，实际并发度往往不理想，可能出现：

- 控制定时器和图像回调互相影响
- `/lowstate` 回调和图像回调争抢执行时机
- 图像回调时间较长时影响控制周期抖动

#### 优化思路

将回调分组：

1. 控制定时器单独一个 callback group
2. 图像订阅单独一个 callback group
3. `/lowstate` 和 `/wirelesscontroller` 可以再按需要拆组

推荐至少做到：

- 控制 loop 独立
- 图像处理独立

这样现有 `2` 线程 executor 才能真正帮助降低控制抖动。

#### 预期收益

- 控制周期更稳定
- 图像回调不会轻易阻塞 control loop
- 在相机频率较高时效果明显

#### 风险与副作用

拆 callback group 后，并发访问会更真实，需要重新审视：

- `image_mutex_`
- `low_state_mutex_`
- `cmd_mutex_`

不过这些锁本来就已经存在，属于低风险改动。

#### 推荐程度

非常推荐，尤其在真机上比在仿真里更值得做。

## 4. 中优先级优化点

### 4.1 目前 RGB 订阅没有参与观测，属于纯额外负担

#### 当前行为

当前真机部署仍然订阅：

- `/camera/color/image_raw`

并在 `rgb_callback()` 中：

- `toCvCopy`
- `clone`
- 保存到 `latest_rgb_image_`

但当前 policy 观测里没有使用 RGB。

#### 优化思路

如果当前部署版本完全不使用 RGB，可以：

1. 直接去掉 RGB 订阅
2. 或改成 debug/可选开关

#### 预期收益

- 减少图像带宽
- 减少一次 `cv_bridge` 转换
- 减少一次 `clone`
- 减少 `image_mutex_` 竞争

#### 风险与副作用

如果后面又想拿 RGB 做显示或多模态输入，就需要重新开回来。

#### 推荐程度

推荐，但优先级低于 obs 和控制链路优化。

---

### 4.2 一个控制周期内多次读取 `low_state_` 和 `cmd_`

#### 当前行为

当前多个 getter 都分别做自己的锁和数据转换，例如：

- `get_base_ang_vel()`
- `get_projected_gravity()`
- `get_dof_pos()`
- `get_dof_vel()`
- `get_command()`

每个函数都：

- 拿一次锁
- 重新组 `std::vector<float>`
- 再 `SimpleTensor::wrap(...)`

#### 优化思路

在 `low_cmd_write()` 开头生成一个轻量 snapshot：

- IMU snapshot
- 关节位置速度 snapshot
- command snapshot

后续 obs term 只读 snapshot，不再分别加锁。

也可以把 snapshot 设计成：

- `struct LowStateSnapshot`
- `struct CommandSnapshot`

每个控制周期只更新一次。

#### 预期收益

- 减少锁次数
- 减少重复格式转换
- 降低控制线程 jitter

#### 风险与副作用

需要改一小层 getter 逻辑，但风险不大。

#### 推荐程度

推荐，收益稳定。

---

### 4.3 `computeObs()` 每帧都在大量 `cat` 和临时分配

#### 当前行为

在 `ManagerBasedEnv::computeObs()` 中：

1. 逐个 term 取 `get_obs()`
2. 放入 `std::vector<SimpleTensor> obs_list`
3. 最后 `SimpleTensor::cat(obs_list)`

而 `SimpleTensor::cat()` 本身会重新分配一整块新内存，把所有数据复制进去。

buffer 层也有类似情况：

- `ObservationBuffer::get_flattened_buffer()`
- `ImageHistoryBuffer::get_history_stack()`

#### 为什么这是系统性开销

这类开销并不只影响视觉 policy，也会影响所有 policy，只是视觉 policy 的 tensor 更大，影响更明显。

#### 优化思路

初始化阶段就计算每个 obs term 的：

- offset
- length

然后每帧直接把每个 term 的结果写入预分配好的 `policcy_obs[id]` 中，不再通过 `cat` 重新拼接。

对于 image history 也可以考虑：

- 预分配一块连续内存
- 用 ring buffer + ordered copy
- 避免 `std::vector<SimpleTensor>` 拼接

#### 预期收益

- 明显减少短周期堆分配
- 对视觉模型收益更大

#### 风险与副作用

这是公共框架层的优化，改动比部署层更深。

需要更仔细验证：

- 各 policy 输入维度是否完全一致
- history 展开顺序是否保持不变
- image stack 顺序是否保持不变

#### 推荐程度

推荐，但属于第二阶段优化。

---

### 4.4 `toVector<float>(action)` 每 tick 复制 action 一次

#### 当前行为

推理完成后，当前做法是：

1. `SimpleTensor action = manager_step(policy_id_)`
2. `const auto act = toVector<float>(action)`
3. 再把 `act[i]` 写入 `low_cmd_`

#### 优化思路

如果 `SimpleTensor` 的数据布局稳定，可以直接读 `action.data_`，避免多做一次 `std::vector<float>` 拷贝。

#### 预期收益

收益不如 obs 侧大，但这是控制线程的固定开销，做掉总是有益的。

#### 风险与副作用

低风险，但优化收益有限。

#### 推荐程度

中等推荐，可跟其他低风险项顺手一起做。

## 5. ONNX Runtime 层的优化点

### 5.1 每次推理都在重新构造 `Ort::Value`

#### 当前行为

在 `Policy::get_action(SimpleTensor obs)` 里，当前每次推理都重新：

- 组输入 shape
- `CreateTensor<float>(...)`
- 调 `session_->Run(...)`
- 再把输出转回 `SimpleTensor`

对于 SRU 模型，还要额外：

- 创建 recurrent input tensor
- 接收 hidden/cell 输出
- 再构造成新的 `SimpleTensor`

#### 优化思路

做输入输出预分配：

1. 预先固定 input/output shape
2. 保留复用的 `Ort::Value`
3. 只更新底层 buffer 数据

如果继续往前走，还可以研究：

- ONNX Runtime I/O Binding
- GPU 路径上的显存绑定

#### 预期收益

- 降低推理时的短时分配
- SRU 模型收益大于纯 MLP

#### 风险与副作用

这个优化对框架理解要求更高。

如果模型 shape 有动态变化，需要额外小心。

#### 推荐程度

推荐，但属于中后期优化。

---

### 5.2 Split SRU 路径中有额外的 `unordered_map<string, SimpleTensor>` 开销

#### 当前行为

在 split SRU ONNX 路径中，推理过程中会大量使用：

- `std::unordered_map<std::string, SimpleTensor> tensors`

并按字符串名字查找：

- `obs`
- `hidden_state`
- `cell_state`
- `next_hidden_state`
- `actions`

#### 优化思路

把字符串驱动的中间张量管理，改成固定索引或结构体字段驱动：

- `obs`
- `hidden_state`
- `cell_state`
- `latent`
- `actions`

也就是用编译期/固定字段代替运行时哈希查找。

#### 预期收益

- 主要收益在 SRU split policy 上
- 对 CPU 侧延迟更友好

#### 风险与副作用

改动比普通 MLP 路径更深，需要确认 exporter signature 不会被破坏。

#### 推荐程度

如果后面发现 `vtm_lstm_sru` / `vtm_gru_sru` 是瓶颈，这项非常值得做。

---

### 5.3 ORT 线程参数目前固定为 `IntraOpNumThreads(1)`

#### 当前行为

当前 `SessionOptions` 里明确设置了：

- `SetIntraOpNumThreads(1)`
- `SetGraphOptimizationLevel(ORT_ENABLE_ALL)`

#### 优化思路

对不同设备做分层调优：

- CPU 路径下测试 `1 / 2 / 4`
- CUDA 路径下确认是否保留 `1` 更稳

还可以按 policy 分开测：

- `motion_mlp`
- `vtm`
- `vtm_lstm_sru`
- `vtm_gru_sru`

#### 预期收益

对 CPU-only 路径可能有效。

对 CUDA 路径未必一定更好，需要实测。

#### 风险与副作用

线程数过大时，Jetson 这类平台反而可能带来抖动。

#### 推荐程度

作为 benchmark 项值得做，但不建议盲调。

## 6. ROS2 通信层的优化点

### 6.1 `/lowstate` 和 `/wirelesscontroller` 更适合“最新值”语义

#### 当前行为

当前这两条订阅都使用普通深度 `10`。

#### 优化思路

如果你的控制语义更偏“只关心最新状态”，那么可以考虑用 fresher-only 的 QoS 策略，减少旧消息堆积。

#### 预期收益

- 降低队列积压
- 减少旧状态进入控制 loop 的概率

#### 风险与副作用

如果链路本身很不稳定，过于激进的 fresher-only 策略可能让有效消息更容易被丢。

#### 推荐程度

中等推荐，需要结合你的实际网络环境和 middleware 表现测试。

---

### 6.2 图像回调内部 `cv_bridge + resize + crop` 仍然是热点

#### 当前行为

深度回调已经只保留最新帧，但每帧仍然要做：

- `cv_bridge::toCvCopy`
- 毫米转米
- resize
- center crop

#### 优化思路

如果后面发现图像线程本身占用过高，可以继续往下挖：

1. 尽量避免不必要的 `clone`
2. 评估是否能使用更轻量的 OpenCV 路径
3. 只在源图尺寸变化时重新计算部分参数

#### 预期收益

这项收益取决于相机频率和原始分辨率，60 Hz 高分辨率时更值得看。

#### 风险与副作用

容易引入图像内容变化、对齐偏差，需要严格对比视觉输入。

#### 推荐程度

中等推荐，优先级低于缓存归一化 tensor。

## 7. 微优化与工程清理项

### 7.1 `record_policy_inference_stats()` 在 monitor 关闭时仍然先加锁

当前实现里一进函数就拿 `perf_stats_mutex_`，然后才判断 monitor 是否开启。

优化思路：

- 先无锁读 `policy_perf_monitor_enabled_`
- 关闭时直接返回

收益很小，但这是个很干净的微优化。

---

### 7.2 日志路径可以继续瘦身

当前日志已经不算多，但一些高频路径如果继续加调试信息，容易影响控制稳定性。

建议原则：

- 控制定时器路径里只保留节流日志
- 图像 stale / recover 用状态切换日志
- 高频统计用窗口聚合打印

---

### 7.3 CRC 计算是固定开销，但不是当前主要矛盾

每次发命令前都会算一次 CRC，这本身是合理且必要的。

目前不建议把精力放在这里，优先级明显低于 obs / image / ORT。

## 8. 推荐落地顺序

如果要控制风险、分阶段推进，建议这样排：

### 第一阶段：低风险高收益

1. 只更新当前 policy 的 obs
2. 当前非视觉 policy 时不刷新视觉 obs
3. 去掉或关闭 RGB 订阅
4. 深度归一化结果改成“新帧到来时更新缓存”

### 第二阶段：控制稳定性优化

1. 拆 callback group
2. 调整 executor 对控制回调和图像回调的并发关系
3. 做 `low_state` / `cmd` snapshot

### 第三阶段：公共框架优化

1. 减少 `SimpleTensor::cat()` 和临时分配
2. 优化 `ObservationBuffer` / `ImageHistoryBuffer`
3. 减少 action/output copy

### 第四阶段：ONNX Runtime 深优化

1. 预分配 `Ort::Value`
2. 优化 split SRU 中间张量管理
3. 评估 I/O binding / CUDA 路径进一步优化
4. benchmark ORT 线程参数

## 9. 建议的验证指标

每做完一项优化，都建议至少记录这几项指标：

### 控制主链路

- 控制循环平均频率
- 控制循环最大周期抖动
- 推理平均耗时
- 推理最大耗时

### 图像链路

- 深度图接收频率
- 深度图从接收到被消费的延迟
- stale 发生频率
- 恢复后首帧延迟

### 系统资源

- CPU 总占用
- 单核 hottest core 占用
- GPU 占用
- 内存占用
- 内存分配抖动

### 模型行为

- policy 切换时是否稳定
- reset 后是否行为一致
- 视觉 policy 在相机短时丢帧后是否正常恢复

## 10. 最终建议

如果只看“投入产出比”，当前最值得先做的三项是：

1. 关闭 `manager_step()` 中“更新所有 policy obs”的默认行为
2. 把视觉 obs 刷新改成按需执行，而不是每个控制 tick 无条件执行
3. 把归一化后的深度观测缓存到图像回调线程，控制线程只读结果

这三项的共同特点是：

- 对当前控制逻辑影响可控
- 不需要立刻重写整个框架
- 对 Jetson 边缘设备通常最有效

如果后面还需要继续压榨性能，再进入：

- callback group 拆分
- `ManagerEnv` 连续内存拼接优化
- ONNX Runtime 输入输出预分配

会更合适。

