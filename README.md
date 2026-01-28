# Pico-Body-Fitter

**Pure Geometric SMPL Fitting for VR Tracking Data**

A lightweight, neural-network-free solution for fitting SMPL body models to Pico VR full-body tracking data. Designed for physics simulation pipelines (Isaac Gym / PHC) with biological constraints and aggressive anti-jitter filtering.

[中文文档](#中文文档) | [English](#english-documentation)

---

## English Documentation

### ✨ Key Features

- **🎯 Pure Geometric**: No neural networks, no heavy weights - just mathematical optimization
- **🦴 Bio-Constraints**: Built-in spine anti-flip, knee hinge locks, and skeletal vector alignment
- **📊 Stability**: Double Butterworth filtering + dynamic temporal smoothing eliminates VR jitter
- **🎮 PHC Ready**: Output format perfectly compatible with Isaac Gym / PHC physics simulation

### 📋 Requirements

```bash
# Core dependencies
numpy>=1.19.0
torch>=1.8.0
scipy>=1.7.0

# SMPL and body models
smplx>=0.1.28

# Visualization
open3d>=0.13.0
matplotlib>=3.3.0

# Data handling
tqdm
```

### 🚀 Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Download SMPL Model

**IMPORTANT**: SMPL models are copyrighted and require a license.

1. Visit https://smpl.is.tue.mpg.de/
2. Register and download **SMPL for Python** (version 1.0.0)
3. Extract and place the following files in `data/smpl_models/`:
   - `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
   - (Optional) `J_regressor_h36m.npy` for H36M joint regressor

Your directory structure should look like:
```
data/smpl_models/
├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
├── J_regressor_h36m.npy (optional)
└── readme.md
```

#### 3. Run Fitting

```bash
# Basic usage
python run_fitting.py data/input_pico/sample.txt

# With custom parameters
python run_fitting.py data/input_pico/sample.txt \
    --iterations 1000 \
    --weight_spine_lock 25.0 \
    --weight_knee_hinge 25.0 \
    --device cuda
```

#### 4. Visualize Results

```bash
python visualize.py data/input_pico/sample.txt data/output/sample_fitted.pkl
```

### 📁 Project Structure

```
Pico-Body-Fitter/
├── src/                      # Core modules
│   ├── filters.py            # Butterworth + Gaussian filtering
│   ├── utils.py              # Data parsing and coordinate transforms
│   ├── smpl_solver.py        # Optimization with biological constraints
│   └── smpl_engine/          # SMPL forward kinematics
├── data/
│   ├── input_pico/           # Input VR tracking data (.txt)
│   ├── output/               # Generated SMPL parameters (.pkl)
│   └── smpl_models/          # SMPL model files (download required)
├── assets/                   # Sample tracking data
├── run_fitting.py            # Main entry point
├── visualize.py              # Visualization tool
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### 🔧 Advanced Usage

#### Coordinate System Fixes

If your tracking data has orientation issues:

```bash
# Disable X-axis mirroring
python run_fitting.py input.txt --no_mirror_fix

# Enable 180° Y-axis rotation
python run_fitting.py input.txt --rotate_180

# Disable root position zeroing
python run_fitting.py input.txt --no_root_zero
```

#### Optimization Parameters

Fine-tune the fitting quality:

```bash
python run_fitting.py input.txt \
    --iterations 1000 \              # More iterations = better fit
    --weight_joints 1.0 \            # Higher = tighter joint fitting
    --weight_spine_lock 30.0 \       # Higher = prevent spine flip
    --weight_knee_hinge 25.0 \       # Higher = prevent lateral knee bend
    --weight_temporal 100.0 \        # Higher = smoother motion
    --smooth_sigma 5.0               # Higher = more aggressive smoothing
```

### 🎯 Technical Approach

#### Three-Stage Filtering Pipeline

1. **Input Filtering** (5 Hz Butterworth)
   - Removes Pico VR tracking noise
   - Applied to raw joint positions

2. **Temporal Smoothness Loss** (Optimization)
   - Penalizes frame-to-frame parameter changes
   - Creates "spring-damper" effect

3. **Output Filtering** (10 Hz Butterworth + Gaussian)
   - Removes optimizer micro-oscillations
   - Final aggressive smoothing

#### Biological Constraints

- **Spine Lock**: Prevents 180° spine flip by enforcing head-pelvis alignment
- **Knee Hinge**: Restricts knee rotation to single axis (prevents lateral bending)
- **Foot Flat**: Keeps feet parallel to ground plane
- **Balance**: Maintains center of mass over support polygon

### 📊 Output Format

The output `.pkl` file contains:

```python
{
    'poses': (N, 72),           # SMPL pose parameters (axis-angle)
    'trans': (N, 3),            # Global translation
    'betas': (10,),             # Body shape parameters
    'mocap_framerate': 30,      # Framerate
    'gender': 'neutral',        # Body model gender
    'target_joints': (N, 24, 3) # Original target positions (for debugging)
}
```

This format is directly compatible with Isaac Gym and PHC simulation pipelines.

### 🐛 Troubleshooting

**Issue**: Left and right sides are swapped
- **Solution**: Use `--no_mirror_fix` flag

**Issue**: Character facing wrong direction
- **Solution**: Use `--rotate_180` flag

**Issue**: Motion is too jittery
- **Solution**: Increase `--weight_temporal` and `--smooth_sigma`

**Issue**: Arms don't match tracking data well
- **Solution**: Increase `--weight_joints` (default: 0.5 → try 1.0)

**Issue**: Spine flips 180° during bending
- **Solution**: Increase `--weight_spine_lock` (default: 20.0 → try 30.0)

### 📄 License

This project is released under the MIT License. However, SMPL models are subject to their own license terms from https://smpl.is.tue.mpg.de/

### 🙏 Acknowledgments

- SMPL body model: https://smpl.is.tue.mpg.de/
- smplx library: https://github.com/vchoutas/smplx
- Inspired by PHC (Perpetual Humanoid Control)

---

## 中文文档

### ✨ 核心特性

- **🎯 纯几何方法**: 无需神经网络，无需重型权重文件 - 纯数学优化
- **🦴 生物学约束**: 内置脊柱防翻转、膝盖铰链锁、骨骼向量对齐
- **📊 极致稳定**: 双重巴特沃斯滤波 + 动态时序平滑，彻底消除 VR 抖动
- **🎮 PHC 就绪**: 输出格式完美适配 Isaac Gym / PHC 物理仿真

### 📋 环境要求

```bash
# 核心依赖
numpy>=1.19.0
torch>=1.8.0
scipy>=1.7.0

# SMPL 人体模型
smplx>=0.1.28

# 可视化
open3d>=0.13.0
matplotlib>=3.3.0

# 数据处理
tqdm
```

### 🚀 快速开始

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 下载 SMPL 模型

**重要提示**: SMPL 模型受版权保护，需要申请许可证。

1. 访问 https://smpl.is.tue.mpg.de/
2. 注册并下载 **SMPL for Python** (版本 1.0.0)
3. 解压后将以下文件放入 `data/smpl_models/` 目录:
   - `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
   - (可选) `J_regressor_h36m.npy` 用于 H36M 关节回归器

目录结构应如下所示:
```
data/smpl_models/
├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
├── J_regressor_h36m.npy (可选)
└── readme.md
```

#### 3. 运行拟合

```bash
# 基础用法
python run_fitting.py data/input_pico/sample.txt

# 自定义参数
python run_fitting.py data/input_pico/sample.txt \
    --iterations 1000 \
    --weight_spine_lock 25.0 \
    --weight_knee_hinge 25.0 \
    --device cuda
```

#### 4. 可视化结果

```bash
python visualize.py data/input_pico/sample.txt data/output/sample_fitted.pkl
```

### 📁 项目结构

```
Pico-Body-Fitter/
├── src/                      # 核心模块
│   ├── filters.py            # 巴特沃斯 + 高斯滤波
│   ├── utils.py              # 数据解析和坐标变换
│   ├── smpl_solver.py        # 带生物学约束的优化器
│   └── smpl_engine/          # SMPL 正向运动学
├── data/
│   ├── input_pico/           # 输入 VR 追踪数据 (.txt)
│   ├── output/               # 生成的 SMPL 参数 (.pkl)
│   └── smpl_models/          # SMPL 模型文件 (需下载)
├── assets/                   # 示例追踪数据
├── run_fitting.py            # 主入口
├── visualize.py              # 可视化工具
├── requirements.txt          # Python 依赖
└── README.md                 # 本文件
```

### 🔧 高级用法

#### 坐标系修正

如果追踪数据存在方向问题:

```bash
# 禁用 X 轴镜像
python run_fitting.py input.txt --no_mirror_fix

# 启用 180° Y 轴旋转
python run_fitting.py input.txt --rotate_180

# 禁用根节点位置归零
python run_fitting.py input.txt --no_root_zero
```

#### 优化参数调整

微调拟合质量:

```bash
python run_fitting.py input.txt \
    --iterations 1000 \              # 更多迭代 = 更好拟合
    --weight_joints 1.0 \            # 更高 = 更紧密的关节拟合
    --weight_spine_lock 30.0 \       # 更高 = 防止脊柱翻转
    --weight_knee_hinge 25.0 \       # 更高 = 防止膝盖侧向弯曲
    --weight_temporal 100.0 \        # 更高 = 更平滑的运动
    --smooth_sigma 5.0               # 更高 = 更激进的平滑
```

### 🎯 技术方法

#### 三阶段滤波流程

1. **输入滤波** (5 Hz 巴特沃斯)
   - 去除 Pico VR 追踪噪声
   - 应用于原始关节位置

2. **时序平滑损失** (优化过程)
   - 惩罚帧间参数变化
   - 创建"弹簧-阻尼器"效果

3. **输出滤波** (10 Hz 巴特沃斯 + 高斯)
   - 去除优化器微振荡
   - 最终激进平滑

#### 生物学约束

- **脊柱锁定**: 通过强制头部-骨盆对齐防止 180° 脊柱翻转
- **膝盖铰链**: 限制膝盖旋转为单轴 (防止侧向弯曲)
- **足部平坦**: 保持足部平行于地面
- **平衡约束**: 维持质心在支撑多边形上方

### 📊 输出格式

输出的 `.pkl` 文件包含:

```python
{
    'poses': (N, 72),           # SMPL 姿态参数 (轴角表示)
    'trans': (N, 3),            # 全局平移
    'betas': (10,),             # 体型参数
    'mocap_framerate': 30,      # 帧率
    'gender': 'neutral',        # 人体模型性别
    'target_joints': (N, 24, 3) # 原始目标位置 (用于调试)
}
```

此格式直接兼容 Isaac Gym 和 PHC 仿真流程。

### 🐛 故障排除

**问题**: 左右侧互换
- **解决方案**: 使用 `--no_mirror_fix` 标志

**问题**: 角色朝向错误
- **解决方案**: 使用 `--rotate_180` 标志

**问题**: 运动过于抖动
- **解决方案**: 增加 `--weight_temporal` 和 `--smooth_sigma`

**问题**: 手臂与追踪数据不匹配
- **解决方案**: 增加 `--weight_joints` (默认: 0.5 → 尝试 1.0)

**问题**: 弯腰时脊柱翻转 180°
- **解决方案**: 增加 `--weight_spine_lock` (默认: 20.0 → 尝试 30.0)

### 📄 许可证

本项目采用 MIT 许可证发布。但是，SMPL 模型受其自身许可条款约束，详见 https://smpl.is.tue.mpg.de/

### 🙏 致谢

- SMPL 人体模型: https://smpl.is.tue.mpg.de/
- smplx 库: https://github.com/vchoutas/smplx
- 灵感来源于 PHC (Perpetual Humanoid Control)

---

**Made with ❤️ for the VR and Physics Simulation Community**
