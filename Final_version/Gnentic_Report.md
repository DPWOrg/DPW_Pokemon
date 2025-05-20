以下是对该遗传算法系统的形式化数学描述：

$$
\renewcommand{\vec}[1]{\mathbf{#1}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\mathrm{Var}}
$$

# 遗传算法形式化描述

## 1. 问题定义

设优化问题为：
$$
\max_{\vec{t} \in \mathcal{T}} F(\vec{t}) = \omega_1 S_t(\vec{t}) + \omega_2 S_s(\vec{t}) + \omega_3 S_m(\vec{t})
$$
其中：
- $\vec{t} = (p_1,...,p_6) \in \mathbb{N}^6$ 表示队伍（6个宝可梦ID）
- $\mathcal{T} = \{\vec{t} | p_i \in \mathcal{P} \setminus \mathcal{E}, p_i \neq p_j\}$ 为可行解空间
- $\mathcal{P}$ 为全体宝可梦集合
- $\mathcal{E} \subset \mathcal{P}$ 为敌方队伍

## 2. 适应度函数分解

### 类型得分
$$
S_t(\vec{t}) = \sum_{\tau \in \Gamma(\vec{t})} \sum_{\epsilon \in \Gamma(\mathcal{E})} \phi(\tau, \epsilon)
$$
其中：
- $\Gamma(\cdot)$ 为队伍类型提取函数
- $\phi: \mathcal{T} \times \mathcal{T} \rightarrow [-1,1]$ 为类型克制函数，定义为：
$$
\phi(\tau,\epsilon) = 
\begin{cases}
2 & \text{双重克制} \\
1 & \text{单克制} \\
0 & \text{无效} \\
-1 & \text{被克制}
\end{cases}
$$

### 属性得分
$$
S_s(\vec{t}) = \frac{1}{6}\sum_{i=1}^6 \vec{s}_i - \frac{\alpha}{2}\sqrt{\frac{1}{5}\sum_{i=1}^6 (\vec{s}_i - \bar{\vec{s}})^2}
$$
其中：
- $\vec{s}_i \in \mathbb{R}^d$ 为第i个宝可梦的标准化属性向量
- $\alpha \in [0,1]$ 为方差惩罚系数

### 模型得分
$$
S_m(\vec{t}) = f_{\theta}(\mathcal{G}(\vec{t}))
$$
其中：
- $\mathcal{G}(\vec{t}) = (V,E)$ 为队伍图表示
  - $V = \{v_i | v_i = MLP(\vec{s}_i)\}_{i=1}^6$ 
  - $E = \{(v_i,v_j) | \forall i < j\}$
- $f_{\theta}$ 为图神经网络参数化函数：
$$
f_{\theta} = \sigma\left(\frac{1}{|V|}\sum_{v \in V} W_2 \cdot \text{ReLU}(W_1 \cdot v)\right)
$$

## 3. 遗传操作

### 种群表示
$$
\mathcal{P}^{(g)} = \{\vec{t}_1^{(g)},...,\vec{t}_N^{(g)}\}, \quad N = |\mathcal{P}^{(g)}| = \text{POP\_SIZE}
$$

### 选择算子（锦标赛选择）
$$
\text{Select}(\mathcal{P}^{(g)}) = \{\arg\max_{\vec{t} \in \mathcal{T}_k} F(\vec{t}) | \mathcal{T}_k \overset{\text{RS}}{\subseteq} \mathcal{P}^{(g)}, |\mathcal{T}_k|=3\}_{k=1}^N
$$
其中RS表示随机采样

### 交叉算子（多点交叉）
对父代$\vec{t}_a = (p_1^a,...,p_6^a)$和$\vec{t}_b = (p_1^b,...,p_6^b)$：
$$
\text{Crossover}(\vec{t}_a, \vec{t}_b) = 
\begin{cases}
\vec{t}_a^{1:i} \oplus \vec{t}_b^{i:j} \oplus \vec{t}_a^{j:6} \\
\vec{t}_b^{1:i} \oplus \vec{t}_a^{i:j} \oplus \vec{t}_b^{j:6}
\end{cases}
$$
其中：
- $i,j \sim U\{1,5\}, i<j$ 为随机交叉点
- $\oplus$ 表示序列拼接
- 需进行去重操作：$\vec{t}' = \text{Unique}(\vec{t})$

### 变异算子
$$
\text{Mutate}(\vec{t}) = \bigotimes_{k=1}^6 M(p_k | \vec{t}_{-k})
$$
其中变异概率：
$$
M(p_k) = \begin{cases}
\text{Uniform}(\mathcal{C}(\vec{t}_{-k})) & \text{w.p. } \mu(g) \\
p_k & \text{otherwise}
\end{cases}
$$
其中：
- $\mu(g) = \mu_0(1 - g/G)$ 为自适应变异率
- $\mathcal{C}(\vec{t}_{-k}) = \{p \in \mathcal{P} | \Gamma(p) \not\subseteq \Gamma(\vec{t}_{-k})\}$ 为类型补全集合

## 4. 算法流程

1. 初始化：
$$
\mathcal{P}^{(0)} \sim \text{Uniform}(\mathcal{T}), \quad |\mathcal{P}^{(0)}| = N
$$

2. 世代迭代 $\forall g \in \{1,...,G\}$：
$$
\mathcal{P}^{(g)} = \text{Mutate}(\text{Crossover}(\text{Select}(\mathcal{P}^{(g-1)})))
$$

3. 精英保留：
$$
\mathcal{P}^{(g)} \leftarrow \mathcal{P}^{(g)} \cup \{\arg\max_{\vec{t}} F(\vec{t}) | \vec{t} \in \mathcal{P}^{(g-1)}\}
$$

4. 终止条件：
$$
g = G \Rightarrow \text{返回 } \{\vec{t} | \vec{t} \in \mathcal{P}^{(G)}, \text{rank}(F(\vec{t})) \leq K\}
$$

## 5. 后处理

最终解集优化：
$$
\mathcal{T}^* = \{\vec{t} | \vec{t} = \arg\max_{\vec{t}' \in \mathcal{T}_i} W(\vec{t}')\}_{i=1}^K
$$
其中：
$$
W(\vec{t}) = \frac{1}{M}\sum_{m=1}^M \mathbb{I}(\text{BattleSim}(\vec{t}, \mathcal{E}) = \text{Win})
$$
为蒙特卡洛胜率估计

该数学描述完整地建立了从问题定义到算法操作的闭合形式化体系，为理论分析和算法改进提供了严格的数学基础。