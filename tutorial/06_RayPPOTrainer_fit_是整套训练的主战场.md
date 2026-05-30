# 第 06 章：`RayPPOTrainer.fit()` 是整套训练的主战场

如果说上一章讲的是“训练系统怎么启动”，那这一章讲的就是：

> 训练一旦启动，Search-R1 到底是怎样完成一轮 rollout、打分、算 advantage、更新模型的？

这个问题的答案主要集中在一个函数里：

- `verl/trainer/ppo/ray_trainer.py:654-867`

也就是：

- `RayPPOTrainer.fit()`

对整个仓库来说，这是最像“总调度中心”的地方。

---

## 1. 为什么这个函数这么重要

因为从训练视角看，Search-R1 的主流程几乎都在这里汇合：

- dataloader 读数据；
- rollout 生成轨迹；
- 调搜索服务；
- 计算 reward；
- 计算 KL；
- 计算 advantage；
- 更新 critic；
- 更新 actor；
- 做验证和存 checkpoint。

如果你把它看懂了，整个项目就不是一堆散碎模块，而是一条完整数据流。

---

## 2. 先看一张总流程图

这一章建议你先记住下面这个大图：

```text
读一个 batch
  -> 复制样本（为多轨迹采样做准备）
  -> 取出 prompt 相关张量
  -> 调 generation_manager 生成带搜索的轨迹
  -> 计算 old_log_probs
  -> 合并回 batch
  -> 计算 ref_log_prob（可选）
  -> 计算 values（PPO 时）
  -> 计算 reward
  -> 计算 token_level_rewards
  -> 计算 advantages / returns
  -> 更新 critic（PPO 时）
  -> 更新 actor
  -> 记录指标
```

Search-R1 和普通 PPO 最关键的差异就藏在中间这一段：

```text
调 generation_manager 生成带搜索的轨迹
```

也就是：
- `ray_trainer.py:725-748`

---

## 3. 进入 `fit()` 之前，trainer 已经准备好了什么

在 `fit()` 运行之前，这个 trainer 已经通过其他函数准备好了三类东西：

### 3.1 数据加载器
来源：
- `ray_trainer.py:372-435`

它会构造：
- `self.train_dataloader`
- `self.val_dataloader`

背后用的是：
- `verl/utils/dataset/rl_dataset.py`

### 3.2 worker 体系
来源：
- `ray_trainer.py:550-622`

它会初始化：
- actor rollout worker
- critic worker（如果是 GAE/PPO）
- reference policy worker（如果启用）
- reward model worker（如果启用）

### 3.3 logger 和 KL controller
来源：
- `ray_trainer.py:348-371`

所以你在看 `fit()` 时，最好把它理解成：

> 所有组件都已经装配好，现在只是在执行训练数据流。

---

## 4. `fit()` 开始时先做什么

函数开头在：
- `ray_trainer.py:661-693`

这里先做了三件事：

### 4.1 初始化全局 step

```python
self.global_steps = 0
```

### 4.2 可选的训练前验证

- `ray_trainer.py:663-670`

如果 `val_before_train=true`，它会先调 `_validate()`。这对理解项目很重要，因为 Search-R1 不是只在训练结束后才评估，而是允许：

- 训练前先测一次基线；
- 训练过程中周期性验证；
- 训练末尾再验证一次。

### 4.3 组装 agent generation 配置

- `ray_trainer.py:675-692`

这里构造了：

```python
gen_config = GenerationConfig(...)
generation_manager = LLMGenerationManager(...)
```

注意这里的字段：

- `max_turns`
- `max_start_length`
- `max_prompt_length`
- `max_response_length`
- `max_obs_length`
- `search_url`
- `topk`

这一步非常关键，因为它把：

- 训练配置中的搜索轮数；
- prompt / response / observation 截断策略；
- 搜索服务地址；

统一塞给了 agent rollout 管理器。

也就是说，从这里开始，trainer 已经准备好“用一个带搜索环境的 LLM agent”去 rollout 了。

---

## 5. 真正训练循环从哪里开始

核心循环在：
- `ray_trainer.py:694-845`

结构是：

```python
for epoch in range(...):
    for batch_dict in self.train_dataloader:
        ...
```

所以从最外层看，它仍然是典型的 epoch / batch 训练范式。

但 Search-R1 的特殊点在于：

> 一个 batch 里的每个样本，不再只是“prompt -> 一次回答”，而是“prompt -> 多轮生成/搜索/观察 -> 最终回答”的轨迹。

---

## 6. 一进 batch，为什么先 `repeat`

看：`ray_trainer.py:701-703`

```python
batch: DataProto = DataProto.from_single_dict(batch_dict)
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
```

这一步第一次看会很迷糊，但其实很重要。

### 6.1 它的直觉是什么

它表示：

- 同一个原始样本，可能要复制成多份；
- 每一份后面会走不同的 rollout 轨迹。

对于 PPO，`n_agent` 通常是 1；
对于 GRPO，`n_agent` 往往大于 1，例如 `train_grpo.sh:62` 里设为 5。

所以这里实际上是在为“同题多轨迹采样”做准备。

---

## 7. 为什么要把 `input_ids / attention_mask / position_ids` 先 `pop` 出去

看：`ray_trainer.py:704-705`

```python
gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
```

这一步的作用可以理解成：

> 先把“生成时真正需要的张量”抽出来，单独送去 rollout。

而原 batch 里还保留着：

- `data_source`
- `reward_model`
- `extra_info`
- `index`

这些非 tensor 或辅助信息。

这种写法的好处是：

- rollout 阶段只关心生成所需张量；
- reward 阶段还能把原始样本信息拿回来用。

这就是 `DataProto` 这种数据组织方式的一个优势：

- 张量流和非张量元信息可以一起走，但在关键阶段又能拆开。

---

## 8. Search-R1 在 trainer 中的关键增强点

从当前代码主线看，最关键的增强点不在于 trainer 被完全重写，而在于：

- trainer 在 rollout 之前先构造了 `GenerationConfig`；
- 然后把生成过程交给 `LLMGenerationManager`；
- 也就是把原本“单轮生成”的位置换成了“可多轮搜索的 agent loop”。

对应位置主要是：
- `ray_trainer.py:675-692`
- `ray_trainer.py:725-748`

这说明 Search-R1 的核心工程思路仍然是：

> 尽量复用 veRL 原有 PPO 数据流，只把 rollout 主体增强成带搜索的多轮 agent 生成。

---

## 9. trainer 是怎样接入 agent loop 的

看：`ray_trainer.py:725-748`

这一段可以拆成五步。

### 第一步：截取初始输入

```python
first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
```

这表示训练不会无脑把全部历史都塞进 rollout，而是保留一个受控长度的起始 prompt。

### 第二步：运行多轮 agent loop

```python
final_gen_batch_output = generation_manager.run_llm_loop(
    gen_batch=gen_batch,
    initial_input_ids=first_input_ids,
)
```

这里就是 Search-R1 最核心的地方。

`run_llm_loop()` 会做：
- 生成；
- 解析 `<search>` 或 `<answer>`；
- 调搜索；
- 注入 `<information>`；
- 再生成；
- 最后返回完整轨迹。

### 第三步：计算 old log probs

```python
output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
final_gen_batch_output = final_gen_batch_output.union(output)
```

这一步是 PPO/GRPO 都需要的，因为后面 policy update 要比较：

- 旧策略下生成这些 token 的 log prob
- 新策略更新后的 log prob

### 第四步：给样本绑定 uid

```python
batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
```

这一步尤其重要。

为什么不是随机 UUID，而是直接复制 `index`？

因为 GRPO 后面真正参与优势计算时，用的是 `uid` 字段；而这里把数据集里的 `index` 复制成 `uid`，就是为了保留“同一个原始问题”的分组语义。

所以更准确地说法是：

> 数据侧先提供 `index`，trainer 再把它提升为本轮训练里真正用于分组的 `uid`。

这里的 `uid` 本质上已经不只是“唯一标识”，更是“分组键”。

### 第五步：把 rollout 结果合回 batch

```python
batch = batch.repeat(...)
batch = batch.union(final_gen_batch_output)
```

这一步之后，batch 终于同时拥有了：

- 原始样本信息；
- 生成结果；
- 搜索轨迹相关张量；
- old log probs。

从这里开始，后面的 reward 和 update 才有足够信息可算。

---

## 10. 为什么还要 `_balance_batch`

看：`ray_trainer.py:753-756`

```python
self._balance_batch(batch, metrics=metrics)
```

这一步是比较工程化的一步，但值得知道它在解决什么问题。

### 10.1 它的目的

不同样本的有效 token 长度差异可能很大。

如果分布式训练时把长样本和短样本分得很不均匀，就会出现：

- 有的 GPU 很忙；
- 有的 GPU 很闲；
- 整体吞吐下降。

所以 `_balance_batch()` 会根据有效序列长度重新排序，让不同 rank 的 token 负载更接近。

### 10.2 这一步带来的一个副作用

代码里自己也提示了：
- `ray_trainer.py:754-755`

> batch 内顺序会被打乱。

所以如果你以后实现依赖“同组样本位置连续”的算法，要特别小心。Search-R1 这里通过 `uid` / `index` 来保证分组语义，而不是依赖样本原始顺序。

---

## 11. reward、KL、advantage 是怎样接上的

这部分在：`ray_trainer.py:766-805`

这是训练主循环中最“RL”的部分。

### 11.1 先算 reference log prob

如果启用了 reference policy：

```python
ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
```

它的作用是后面估计 KL，用来约束当前策略不要偏离太快。

### 11.2 PPO 时再算 values

如果当前是 GAE/PPO 路线：

```python
values = self.critic_wg.compute_values(batch)
```

这说明 critic 只在 PPO/GAE 路线里是必须的。

### 11.3 算 reward

```python
reward_tensor = self.reward_fn(batch)
batch.batch['token_level_scores'] = reward_tensor
```

这里的 `reward_fn` 默认就是上一章讲的 `RewardManager`。

它会根据最终答案是否命中 ground truth 给 outcome reward。

### 11.4 处理 KL

如果不是直接在 actor loss 里显式使用 KL：

```python
batch, kl_metrics = apply_kl_penalty(...)
```

那就会把 KL 融入 `token_level_rewards` 里。

### 11.5 计算 advantage

```python
batch = compute_advantage(batch, adv_estimator=...)
```

这一步会根据配置决定：

- 走 GAE；
- 还是走 GRPO。

这也是 PPO 和 GRPO 在训练主线上真正分叉的地方。

---

## 12. 为什么 PPO 会先更新 critic，再更新 actor

看：`ray_trainer.py:807-822`

### 12.1 更新 critic

```python
critic_output = self.critic_wg.update_critic(batch)
```

critic 的任务是：
- 拟合 returns / values；
- 为 PPO 的 advantage 学习提供稳定估计。

### 12.2 再更新 actor

```python
actor_output = self.actor_rollout_wg.update_actor(batch)
```

actor 的任务才是：
- 用 advantages 驱动策略改进。

所以这条顺序本质上是：

1. 先把“价值估计器”更新好；
2. 再据此更新策略。

如果是 GRPO，则不会走 critic 那条分支，因为它不依赖单独 value model。

---

## 13. `state_masking` 在这里扮演什么角色

看：`ray_trainer.py:818-820` 以及 `ray_trainer.py:854-867`

当配置里启用了：

- `actor_rollout_ref.actor.state_masking=true`

trainer 会调用 `_create_loss_mask()`：

```python
loss_mask = batch.batch['info_mask'][:, -response_length:]
batch.batch['loss_mask'] = loss_mask
```

这一步和 Search-R1 的一个核心设计相关：

> 搜索返回的 `<information>` 属于环境 observation，不一定应该像模型自己生成的 token 一样参与 loss。

所以这里会构造一个 `info_mask / loss_mask`，控制哪些 token 真正参与策略学习。

这也是 Search-R1 与普通纯文本 PPO 的一个关键差异。

---

## 14. 这一章里你最该盯住的变量

如果你在 `fit()` 里打断点，我建议优先看这些变量：

```python
p batch.batch.keys()
p batch.non_tensor_batch['index'][:5]
p gen_batch.batch['input_ids'].shape
p final_gen_batch_output.batch.keys()
p batch.batch['token_level_scores'].shape
p batch.batch['token_level_rewards'].shape
p batch.batch['advantages'].shape
p metrics.keys()
```

这些变量会帮助你把整条主线串起来。

---

## 15. 这一章最重要的理解不是细节，而是“插入点”

我特别希望你抓住一个事实：

> Search-R1 没有推倒重写 PPO trainer，它是把“多轮搜索 agent rollout”插到了 trainer 的 rollout 位置上。

也就是说，整个工程的核心创新不是：

- 改一个小 loss；
- 改一个数据结构；

而是：

- 把 rollout 从“单轮文本生成”变成“生成—搜索—观察—再生成”的交互循环；
- 然后让后面的 PPO/GRPO 数据流照常接上。

这就是为什么下一章我们必须专门拆开 `generation.py` 来讲。

---

## 16. 本章小结

这一章最关键的结论有五个：

1. **`RayPPOTrainer.fit()` 是 Search-R1 训练主流程的总调度中心。**
2. **Search-R1 与普通 PPO 的核心差异，集中在 rollout 这一段。**
3. **`run_llm_loop()` 把单轮生成升级成了多轮 agent 交互。**
4. **reward、KL、advantage、actor/critic update 在 rollout 之后按标准 RL 数据流接上。**
5. **GRPO 的关键分组语义依赖 `index -> uid` 这条链。**

下一章，我们就进入这个最核心的增量文件：`search_r1/llm_agent/generation.py`。