# POPPY 与 Diffractio 使用速览

> 说明：当前环境无法直接访问 GitHub（`curl` 返回 403），以下是基于两者常见工作流整理的上手方式。

## 1) POPPY（spacetelescope/poppy）

POPPY 是一个以**傅里叶光学传播**为核心的 Python 库，常用于望远镜/成像系统 PSF 建模。

### 安装

```bash
pip install poppy
```

常与这些库一起安装：

```bash
pip install numpy scipy matplotlib astropy
```

### 最小示例：构建一个简单光学系统并计算 PSF

```python
import poppy
import matplotlib.pyplot as plt

# 定义光学系统
osys = poppy.OpticalSystem()
osys.add_pupil(poppy.CircularAperture(radius=1.0))
osys.add_detector(pixelscale=0.01, fov_arcsec=2.0)

# 计算点扩散函数
psf = osys.calc_psf(wavelength=1e-6)  # 1 micron

# 可视化
poppy.display_psf(psf)
plt.show()
```

### 典型使用流程

1. 定义 pupil（口径、遮挡、相位项等）。
2. 配置 detector（像元尺度、视场）。
3. 调用 `calc_psf(...)` 做单波长或多波长传播。
4. 用 `display_psf` / `display_ee` 等函数分析结果。

---

## 2) Diffractio（optbrea/diffractio）

Diffractio 更偏向**标量/矢量衍射场**仿真，支持 1D/2D/3D 场与多种传播算法。

### 安装

```bash
pip install diffractio
```

常见配套依赖：

```bash
pip install numpy scipy matplotlib
```

### 最小示例：平面波 + 圆孔 + 传播

```python
import numpy as np
from diffractio import um, mm
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY

# 网格
x = np.linspace(-0.5*mm, 0.5*mm, 1024)
y = np.linspace(-0.5*mm, 0.5*mm, 1024)

# 光源
u0 = Scalar_source_XY(x=x, y=y, wavelength=0.633*um)
u0.plane_wave(A=1)

# 掩模（圆孔）
mask = Scalar_mask_XY(x=x, y=y, wavelength=0.633*um)
mask.circle(r0=(0, 0), radius=0.1*mm)

# 通过掩模并传播（示意）
u1 = u0 * mask
u2 = u1.RS(z=50*mm)  # Rayleigh-Sommerfeld 传播

u2.draw(kind='intensity')
```

### 典型使用流程

1. 先定义坐标网格与波长（单位常用 `um/mm`）。
2. 建立 source（平面波、高斯光束等）。
3. 叠加 mask/DOE/透镜等元件。
4. 选传播算法（如 `RS`、`Fresnel`、`FFT` 相关方法）。
5. 用 `draw()` 或导出数组做后处理。

---

## 3) 二者怎么选

- 目标是**望远镜/成像系统 PSF**、天文仪器链路：优先 POPPY。
- 目标是**通用衍射场传播、掩模/DOE 设计验证**：优先 Diffractio。
- 实践中可组合：前段波前/系统级用 POPPY，局部衍射结构细节用 Diffractio。

## 4) 常见踩坑

1. **采样不足**：网格点数太小会引起混叠，先增加采样点和视场。
2. **单位混乱**：`m/mm/um` 混用最容易出错，建议统一单位后再传参。
3. **传播距离不匹配**：近场/远场算法要匹配物理场景。
4. **性能问题**：优先从较小网格验证流程，再放大到高分辨率计算。

## 5) 你给的 MATLAB 三模型代码，能否用 Python 跑？

可以。仓库里已补了一个等价的 Python 版本：`docs/dual_doe_three_models_python.py`，实现了这几块核心模型：

- `propagate_asm_bandlimited`（ASM）
- `propagate_fftrs_transfer`（FFT-RS 传递函数）
- `propagate_fresnel_transfer`（Fresnel 传递函数）
- `propagate_spectral_to_custom_grid`（频域传播 + 自定义焦面网格精确逆傅里叶）

运行方式：

```bash
python docs/dual_doe_three_models_python.py \
  --doe1 "圆形DOE1_入射场整形.dat" \
  --doe2 "圆形DOE2_精细调控.dat"
```

说明：

1. 该脚本是 **NumPy CPU 版**，便于先验证模型一致性。
2. 若要加速，可后续替换为 CuPy（GPU）或将矩阵求值块改为分块计算。
3. 目前默认打印关键误差指标（如 FFT-RS/Fresnel 相对 ASM 的焦面归一化强度误差）。
