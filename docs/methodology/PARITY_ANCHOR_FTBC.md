# Parity-Anchor FTBC（PA-FTBC）

## 1. 动机

Temporal-LR FTBC能够压缩逐时间步偏置，但部署前需要：

1. 按阈值归一化所有SignedIF层；
2. 跨层拼接偏置矩阵；
3. 执行SVD并保存共享时间基；
4. 每个时间步用稠密rank-4矩阵乘法合成偏置。

对四套正式协议的校准验证数据进行诊断后发现，学习到的主要时间结构并不是一般的平滑低秩曲线，而是：

- 第一个时间步具有独立的大幅瞬态；
- 第二个时间步仍具有独立瞬态；
- 后续时间步主要由稳定均值和奇偶交替残差组成。

因此没有必要学习任意的共享时间基。

## 2. 方法

设某层Full-FTBC教师偏置为：

\[
B_l\in\mathbb R^{T\times C_l}.
\]

构造固定解析基：

\[
P^{(T)}=[e_0,e_1,p_{\mathrm{tail}},p_{\mathrm{parity}}]
\in\mathbb R^{T\times4},
\]

其中：

- \(e_0\)：只在\(t=0\)为1；
- \(e_1\)：只在\(t=1\)为1；
- \(p_{\mathrm{tail}}\)：在\(t\ge2\)为1；
- \(p_{\mathrm{parity}}\)：在\(t\ge2\)按\(+1,-1,+1,-1,\ldots\)交替。

每层只拟合四个通道向量：

\[
A_l=(P^{(T)})^+B_l\in\mathbb R^{4\times C_l}.
\]

部署时：

\[
\widehat B_l(t)=
\begin{cases}
A_l[0],&t=0,\\
A_l[1],&t=1,\\
A_l[2]+(-1)^tA_l[3],&t\ge2.
\end{cases}
\]

对本实验使用的偶数时间步，尾段常量项和奇偶项正交；实现仍使用固定4列矩阵的伪逆，以保持公式对一般时间步有效。

## 3. 与Temporal-LR的差异

PA-FTBC不需要：

- SVD；
- 跨层拼接；
- 阈值归一化；
- 学习或存储共享时间基；
- 稠密rank-4逐时间步合成。

它只需要每层四个通道向量，并在尾段进行一次向量加法或减法。

## 4. 存储与计算

记所有SignedIF层通道数之和为\(C=\sum_l C_l\)。当\(T>4\)时：

\[
P_{\mathrm{Full}}=TC,
\]

\[
P_{\mathrm{Temporal}}=4T+4C,
\]

\[
P_{\mathrm{PA}}=4C.
\]

以非零时间基项数量作为偏置合成MAC等价值：

\[
M_{\mathrm{Temporal}}=4TC,
\]

\[
M_{\mathrm{PA}}=(2T-2)C.
\]

因此T=8/16/32时，PA相对Temporal分别减少56.25%、53.125%和51.5625%的偏置合成MAC等价值。

## 5. 低时间步与门控

当\(T\le4\)时，PA-FTBC直接回退Full-FTBC，保证没有压缩损失。SNM使用现有A-SNM规则，并为QCFS、Full、Temporal和PA四个家族分别在校准验证集上冻结开关；测试集不参与方法或门控选择。

## 6. 实验口径

正式实验覆盖：

- CIFAR-10/ResNet20 QCFS-L4；
- CIFAR-10/VGG16 QCFS-L8；
- CIFAR-100/ResNet20 QCFS-L8；
- CIFAR-100/VGG16 QCFS-L8；
- \(T\in\{1,2,4,8,16,32\}\)。

正式报告由`scripts/experiments/run_pa_ftbc_asnm_ablation.py`直接生成。早期等距结点和时间幂映射尝试只保存在`docs/archive/experiments/linear_knot/`，没有参与正式测试集方法选择。
