# 第 09 章：PPO、GAE 与 GRPO 在代码里的实现

这一章是整套讲义里 RL 味道最重的一章。

但我会尽量沿着你现在的背景来讲：

- 你已经懂深度学习；
- 你知道 MDP、Q-learning、SARSA 这些词；
- 但对策略梯度、PPO、GAE、GRPO 还不够熟。

所以这章不会从抽象公式堆开始，而是围绕一个问题：

> Search-R1 在得到最终答案对不对这个奖励之后，究竟是怎样把它变成可训练的策略更新信号的？

核心代码主要在：

- `verl/trainer/ppo/core_algos.py:70-275`
- `verl/trainer/ppo/ray_trainer.py:123-154`

---

## 1. 先记住三个层次

你可以把这一章拆成三层来理解：

### 第一层：reward
系统先知道这条回答最终得了多少分。

### 第二层：advantage / return
系统把这个“最终得分”转成更适合优化的训练信号。

### 第三层：loss
系统再用这些信号更新：
- actor
- critic（如果有）

所以主线可以写成：

```text
outcome reward
  -> token-level reward representation
  -> advantages / returns
  -> policy loss / value loss
```

---

## 2. 为什么不能直接用“最终答对/答错”来更新所有 token

这一步非常关键。

假设一条轨迹最后得分为 1，另一条得分为 0。

如果你粗暴地说：

- 第一条轨迹里所有 token 都是好 token；
- 第二条轨迹里所有 token 都是坏 token；

那会有两个问题：

### 2.1 credit assignment 太粗糙

- 哪一步 search 是真正关键的？
- 哪一步 think 是冗余的？
- 最终 answer 对，到底是前面哪一段贡献最大？

直接平均分配很难表达这些差异。

### 2.2 训练容易不稳定

尤其是在长序列、多轮工具调用的场景下，如果信号过于粗糙，actor 很容易学得非常 noisy。

所以 PPO / GAE / GRPO 的任务，本质上就是：

> 把“最终 outcome reward”加工成更适合训练的中间信号。

---

## 3. PPO 路线：先看 GAE 在这里扮演什么角色

在 Search-R1 的 PPO 路线中，最关键的配置是：

- `train_ppo.sh:41` → `algorithm.adv_estimator=gae`

对应 trainer 分支在：
- `ray_trainer.py:126-139`

也就是说，PPO 路线下 advantage 主要由：

- `compute_gae_advantage_return(...)`

来算。

---

## 4. `compute_gae_advantage_return()` 怎么理解

函数位置：
- `core_algos.py:70-107`

### 4.1 输入是什么

它需要：

- `token_level_rewards`
- `values`
- `eos_mask`
- `gamma`
- `lam`

也就是说，它既依赖：
- reward；
- 也依赖 critic 给出的 value 估计。

### 4.2 代码直觉

它从后往前遍历序列：

```python
for t in reversed(range(gen_len)):
    nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
```

你可以把它理解成：

- 先看当前 token 的即时 reward；
- 再看后续价值；
- 再结合 value baseline 去估计“这一位置到底比预期好多少”。

### 4.3 输出是什么

```python
advantages, returns
```

其中：
- `returns = advantages + values`

这正是 PPO 里最常见的 actor-critic 结构。

### 4.4 这对 Search-R1 有什么意义

在 Search-R1 里，很多奖励其实最终才出现，例如：
- 最终 `<answer>` 命中 ground truth。

GAE 的作用就是：

> 把这种晚出现的奖励更平滑地向前传播，而不是简单粗暴地平铺到所有 token。

---

## 5. 为什么 PPO 路线需要 critic

这个问题很重要。

从代码上看：

- `ray_trainer.py:566-576`
- `ray_trainer.py:773-776`
- `ray_trainer.py:807-812`

都说明：
- GAE/PPO 路线会启用 critic；
- critic 负责算 `values`；
- 后面还会更新 critic 本身。

### 5.1 直觉上怎么理解 critic

你可以把 critic 理解成：

> 一个“当前状态值多少钱”的估计器。

有了它，你才能问：

- 当前真实结果比预期更好还是更差？

这就是 advantage 的核心。

### 5.2 对你来说，一个好用的记忆方式

- actor 决定“怎么做”；
- critic 估计“做到这里值多少”。

PPO 的稳定性，很大程度上来自这两者配合。

---

## 6. GRPO 路线：为什么它在优势计算上看起来不需要 critic

在 Search-R1 的 GRPO 路线中，入口配置是：

- `train_grpo.sh:41` → `algorithm.adv_estimator=grpo`

而在 trainer 里会走：
- `ray_trainer.py:140-151`

对应算法函数是：
- `core_algos.py:110-155`

也就是：
- `compute_grpo_outcome_advantage(...)`

---

## 7. `compute_grpo_outcome_advantage()` 在做什么

先看它的输入：

- `token_level_rewards`
- `eos_mask`
- `uid`

注意这里没有 `values`。

这已经说明：

> GRPO 这条实现路线在优势计算上不依赖单独的 critic 价值估计。

这里我特意加一句限定：这是在说 **advantage 的构造方式**。从训练入口装配角度，代码里仍然有统一的 worker / role 组织逻辑；不要把它误解成“整个 GRPO 代码路径里完全不存在 critic 相关装配”。

### 7.1 第一步：先把 outcome reward 压成一个序列分数

代码：

```python
non_zero_mask = (token_level_rewards != 0)
scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
```

因为 Search-R1 的 outcome reward 常常只在最后一个 token 位置非零，所以这里其实就是：

- 把每条 response 的最终得分抽出来。

### 7.2 第二步：按 `uid` 分组

代码：

```python
for i in range(bsz):
    id2score[index[i]].append(scores[i])
```

也就是说：

- 同一个 `uid` 的多条 rollout 属于同一题；
- 而这个 `uid` 在 trainer 中通常是由样本 `index` 复制而来；
- 它们会被放在一组里比较。

### 7.3 第三步：做组内标准化

代码：

```python
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
```

这一步的直觉是：

> 不是单独看“这条轨迹绝对好不好”，而是看“它在同题多条轨迹里相对好不好”。

这就是 GRPO 的核心味道。

### 7.4 第四步：再铺回 token 维度

```python
scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
```

这里会把组内标准化后的分数扩展到整个 response 长度上。

所以从实现上说，GRPO 在这里更像：

- 先对每条轨迹给一个组内相对分；
- 再把这个相对分变成 token 级训练信号。

---

## 8. 你应该怎样直觉地区分 GAE 和 GRPO

我建议你记下面这张对照表：

| 维度 | PPO / GAE | GRPO |
| --- | --- | --- |
| 是否需要 critic | 需要 | 不需要 |
| advantage 来源 | reward + value baseline | 同题多轨迹组内比较 |
| 更像什么 | actor-critic 路线 | relative ranking 路线 |
| 对分组键（数据侧 `index`，训练侧 `uid`）的依赖 | 低 | 高 |

这张表对理解 Search-R1 很有用，因为它直接解释了：

- 为什么 `train_ppo.sh` 和 `train_grpo.sh` 差别这么大；
- 为什么 GRPO 要特别关注 `n_agent` 和分组键（数据侧 `index`，训练侧 `uid`）。

---

## 9. policy loss 是怎么计算的

函数：
- `core_algos.py:163-194`

也就是：
- `compute_policy_loss(...)`

### 9.1 核心思想

代码里最关键的是：

```python
ratio = torch.exp(log_prob - old_log_prob)
pg_losses = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
```

如果用直觉解释就是：

- 新策略希望让高 advantage 的动作更可能；
- 让低 advantage 的动作更不可能；
- 但更新不能太猛，所以用 clip 限制 ratio。

### 9.2 这和 PPO 的名字有什么关系

PPO 全名是：
- Proximal Policy Optimization

其中 “proximal” 的核心意思就是：

> 新策略不要离旧策略太远。

代码里的 `clip` 正是在做这件事。

---

## 10. value loss 是怎么计算的

函数：
- `core_algos.py:216-239`

也就是：
- `compute_value_loss(...)`

它的目标是让 critic 的 value 预测尽量接近 returns。

其中也用了 clipped value 版本：

```python
vpredclipped = clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
```

这和 PPO 的整体思路一致：

- 不仅 policy 更新要稳；
- value 更新也尽量别震荡太大。

---

## 11. KL penalty 在这里起什么作用

相关函数：
- `core_algos.py:242-275`
- `ray_trainer.py:91-120`

KL 的本质作用是：

> 约束当前策略不要偏离 reference policy 太快。

在 Search-R1 场景里，这个约束尤其有意义，因为：

- 我们既想让模型学会搜索；
- 又不想它训练几步后语言行为彻底崩掉。

### 11.1 `low_var_kl`
你会在 GRPO 脚本里看到：
- `kl_loss_type=low_var_kl`

对应实现是：
- `core_algos.py:264-268`

这是一种更稳定的 KL 近似形式。

从实操上，你可以先不深究公式推导，只要知道：

- 它是在帮助训练稳定；
- 是 PPO/GRPO 常见的“别偏太快”机制。

---

## 12. Search-R1 里 PPO / GRPO 的真正区别，不只在公式

这点很重要。

表面上看，区别似乎只是：
- 一个用 GAE；
- 一个用 GRPO。

但在 Search-R1 里，它还体现在更早的地方：

### 12.1 采样结构不同

GRPO 通常要求：
- 同一问题采样多条轨迹；
- 这就是 `n_agent > 1` 的意义。

### 12.2 数据字段的重要性不同

PPO 路线中分组键没那么显眼；
GRPO 路线中，数据侧的 `index` 会在训练侧转成 `uid`，这条分组键几乎是核心。

### 12.3 更新模块不同

PPO：
- actor + critic

GRPO：
- 更多依赖 actor + relative score

所以你最好不要把两者理解成“只是换了一个 advantage 函数”，而要理解成：

> 它们连 rollout 组织方式和样本分组语义都一起变了。

---

## 13. 结合 Search-R1 场景，怎么直觉理解这两条路线

你可以想象同一个问题：

> 模型可能会搜，也可能不搜；搜的 query 可能好，也可能差；最后答案可能对，也可能错。

### PPO/GAE 的思路更像：

- 我有一个 critic 来估计当前序列值多少钱；
- 再根据实际结果和预期差距来更新策略。

### GRPO 的思路更像：

- 同一题我采样 5 条轨迹；
- 比较它们谁更好；
- 相对更好的轨迹被鼓励，相对更差的被压下去。

对搜索型 agent 而言，GRPO 这种“同题多样本相对比较”其实非常自然。

---

## 14. 这一章最值得调试的变量

如果你要配合 `pdb` 理解这一章，我建议重点看：

### PPO/GAE 路线
```python
p token_level_rewards.shape
p values.shape
p advantages.shape
p returns.shape
```

### GRPO 路线
```python
p scores
p batch.non_tensor_batch['uid'][:10]
p id2score
p advantages.shape
```

### actor / critic 更新前
```python
p batch.batch['old_log_probs'].shape
p batch.batch['advantages'].shape
p batch.batch['returns'].shape
```

这些变量足以把“reward -> advantage -> loss”的主链路看清。

---

## 15. 本章小结

这一章最关键的结论有六个：

1. **PPO / GAE 的核心是：reward + critic value -> advantage / return。**
2. **GRPO 的核心是：同题多轨迹组内比较 -> relative advantage。**
3. **PPO 需要 critic，GRPO 的这条实现不依赖 critic。**
4. **数据侧 `index`、训练侧 `uid` 这条分组键链路，是 GRPO 成立的关键。**
5. **policy loss 的 clip 机制是 PPO 稳定性的核心。**
6. **在 Search-R1 中，PPO / GRPO 的差别不仅体现在公式，也体现在 rollout 组织方式上。**

下一章，我们就把搜索系统本身讲透：检索服务、语料、索引，以及为什么 Search-R1 能这么灵活地替换搜索引擎。