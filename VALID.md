复现该论文的完整流程涉及模型构建、参数设置、数值实现和结果分析等多个方面。以下是详细的复现指南，包含所有必要的公式、参数和步骤：

一、模型构建与仿真设置

1. 仿真几何与初始条件

• 倾斜渠道坡度：θ = 20°

• 泥石流体积：500 m³（假设渠道宽度10 m，二维平面应变模型）

• 初始长深比：L_D / H_D ≈ 8

• 双刚性屏障高度：H_B = 2 × h_flow（h_flow为自由流动条件下的最大流深）

2. 网格与材料点生成

• 背景网格尺寸与屏障高度比：0.04（经网格敏感性分析确定）

• 每网格单元初始生成16个材料点（固体和流体相各一套）

• 采用二维平面应变假设

3. 边界条件

• 流体相：自由滑移边界（法向速度为零）

• 固体相：Coulomb摩擦定律，基底摩擦系数 μ_bed = 0.4

二、控制方程与本构模型

1. 动量方程（两相耦合）

固体相：
\bar{\rho}_s \frac{D v_s}{D t} = \bar{\rho}_s g + \nabla \cdot \sigma' - f_d - \phi \nabla p_f
流体相：
\bar{\rho}_f \frac{D v_f}{D t} = \bar{\rho}_f g + \nabla \cdot \mathbf{T}_f + f_d - (1 - \phi) \nabla p_f
其中：
• \bar{\rho}_s = \phi \rho_s, \bar{\rho}_f = (1 - \phi) \rho_f

• f_d 为相间拖曳力，由（22）式计算

2. 流体相本构（不可压缩牛顿流体）

• 不可压缩条件：\nabla \cdot v_f = 0

• 剪应力张量：

\mathbf{T}_f = \eta_f \left(1 + \frac{5}{2} \phi \right) \left[ \nabla v_f + (\nabla v_f)^T \right]

3. 固体相本构（剪切率相关Drucker-Prager）

屈服面：
• 压缩屈服面：f_{\text{compaction}} = g(\phi) p' - (a \phi)^2 \left[ (\dot{\gamma}^p)^2 d^2 \rho_s + 2 \eta_f \dot{\gamma}^p \right]

• 剪切屈服面：f_{\text{shear}} = \sqrt{J_2} - \mu_p p'

摩擦系数：
\mu_p = \mu_1 + \frac{\mu_2 - \mu_1}{1 + b / I_m} + \frac{5}{2} \left( \frac{\phi I_v}{a I_m} \right)
塑性势函数：
P_{\text{shear}} = \sqrt{J_2} - \beta p', \quad \beta = K_4 (\phi - \phi_{\text{eq}})

4. 相间拖曳力

f_d = \frac{18 \phi (1 - \phi) \eta_f}{d^2} \hat{F} (v_s - v_f)
其中 \hat{F} 为固体分数和雷诺数的函数（见Van der Hoef等，2005）。

三、参数设置（来自Table 1）

1. 两相模型参数

参数 符号 值 单位

固体密度 ρ_s 2650 kg/m³

流体密度 ρ_f 1000 kg/m³

流体动力粘度 η_f 0.001 Pa·s

固体颗粒直径 d 1 mm

杨氏模量 E 10 MPa

泊松比 ν 0.3 -

静态摩擦系数 μ₁ 0.49 -

极限摩擦系数 μ₂ 1.4 -

临界固体体积分数 φ_m 0.56 -

dilatancy参数 K₄ 4.7 -

μ(I)参数 a 1.23 -

μ(I)参数 b 0.31 -

基底摩擦系数 μ_bed 0.4 -
2. 等效流体模型参数
参数 符号 值 单位

等效密度 ρ_eq 1924 kg/m³

等效摩擦系数 μ_eq 0.27 -

等效基底摩擦系数 μ_bed_eq 0.19 -

四、仿真流程与步骤

1. 自由流动仿真（无屏障）

• 目的：获取自由流动条件下的流深（h_flow）和流速（v_flow）

• 计算Froude数：$Fr = \frac{v_{\text{flow}}}{\sqrt{\g\
 h_{\text{flow}} \cos \theta}}$

• 根据目标Fr（2、4、6）确定第一道屏障位置

2. 单屏障冲击仿真

• 模拟泥石流冲击第一道屏障

• 记录冲击力、溢出速度（v_launch）、抛出角度（θ_launch）

3. 双屏障仿真

• 改变屏障间距 L/x_i（1.5~5.0）

• 分析落地后流体的重新加速、流体化比率变化、第二道屏障冲击力



五、后处理与关键输出量

1. 流体化比率（λ）

\lambda = \frac{p_{\text{bed}}}{\sigma_{\text{bed}}} = \frac{F_{\text{bed}}^{\text{fluid}}}{F_{\text{bed}}^{\text{fluid}} + F_{\text{bed}}^{\text{solid}}}

2. 冲击力计算

F = \alpha \rho v^2 h + \frac{1}{2} k h^2 \rho \g\

其中α、k取1（Ng等2020a推荐值）

3. 落地距离与速度

• 落地距离：$x_i = \frac{v_{\text{launch}}^2}{\g\ \cos \theta} \left( \tan \theta + \sqrt{\tan^2 \theta + \frac{2 \ g\
 H_B}{v_{\text{launch}}^2 \cos \theta}} \right) + H_B \tan \theta$

• 落地后速度：v_i = R v_r \cos \theta_{\text{land}}（R=1为无耗散假设）

六、验证与标定

1. 实验数据对比

• 干砂、清水、砂水混合物冲击刚性屏障实验

• 对比冲击力时程曲线（Fig. 5）和运动学过程（Fig. 4）



2. 网格敏感性分析

• 网格尺寸与屏障高度比从0.04降至0.02，峰值冲击力仅变化1%（Fig. 13）



七、参数研究方案（Table 2）

仿真ID Fr L/x_i 说明

Fr2S_T/Fr2S_E 2 - 单屏障

Fr2D1.5_T 2 1.5 双屏障，不同间距

...（其他组合） ... ... ...

Fr6D5.0_T 6 5.0 最大间距

八、注意事项

1. 采用C++或Python实现MPM算法，需支持两相耦合；
2. 不可压缩流体相需使用算子分裂法求解压力；
3. 剪切率相关本构需迭代更新塑性应变；
4. 落地过程中的流体化比率变化（Fig. 8）和剪切行为（Fig. 9）是分析重点。





通过以上步骤和参数，可完整复现论文中两相MPM模型对泥石流冲击双屏障的模拟结果。