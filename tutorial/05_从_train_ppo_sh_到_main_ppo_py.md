# 第 05 章：从 `train_ppo.sh` 到 `main_ppo.py`

这一章开始，我们正式进入训练主线。

如果说前几章是在建立 Search-R1 的整体认知，那么这一章要解决的问题是：

> 当你执行 `bash train_ppo.sh` 时，程序到底是怎样一步步走到 Python 训练入口里的？

对你这样的读者来说，这一章尤其重要，因为你已经能看懂 PyTorch 代码，但对 RL 训练框架的“入口组织方式”还不够熟。这里最容易迷糊的一点是：

- shell 脚本里改了很多参数；
- Python 入口里又有一套默认配置；
- 真正执行时，到底以谁为准？

这一章就是把这个配置流和启动流讲清楚。

---

## 1. 先看总图：训练是怎么启动的

PPO 训练主线可以先记成下面这条链：

```text
train_ppo.sh
  -> python -m verl.trainer.main_ppo
  -> main(config)
  -> main_task(config)
  -> 构造 RewardManager
  -> 构造 RayPPOTrainer
  -> trainer.init_workers()
  -> trainer.fit()
```

GRPO 的链路几乎完全一样，只是配置不同：

```text
train_grpo.sh
  -> python -m verl.trainer.main_ppo
  -> 后续主线相同
```

这说明一个非常重要的事实：

> Search-R1 没有单独写一个 `main_grpo.py`。PPO 和 GRPO 共用同一个训练入口，区别主要体现在配置项上。

---

## 2. `train_ppo.sh` 在做什么

先看文件：`train_ppo.sh:1-90`。

这个脚本不是“训练逻辑本身”，而是**训练配置的集中覆盖层**。你可以把它理解成：

- 负责把实验参数写清楚；
- 负责把默认 yaml 配置覆盖掉；
- 负责拼成最终的训练命令。

### 2.1 脚本里最值得你先看的部分

#### 第一块：基础环境变量

例如：

- `train_ppo.sh:1-2`
- `train_ppo.sh:6-22`

这里定义了：

- 用哪些 GPU；
- 数据目录；
- 基座模型；
- 实验名。

其中最值得注意的是：

```bash
export BASE_MODEL='meta-llama/Llama-3.2-3B'
export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-em
```

也就是说，这个脚本把“模型选择”和“实验标识”放在了最外层。

#### 第二块：核心算法选择

最关键的是：

- `train_ppo.sh:41`

```bash
algorithm.adv_estimator=gae
```

这行实际上决定了：

- 你走的是 PPO/GAE 路线；
- 后面会启用 critic；
- advantage 会用 GAE 计算。

#### 第三块：检索相关配置

例如：

- `train_ppo.sh:87-89`

```bash
max_turns=2 \
retriever.url="http://127.0.0.1:8000/retrieve" \
retriever.topk=3 \
```

这里非常关键，因为它告诉你：

- 训练过程默认允许最多 2 轮搜索；
- 搜索服务是独立 HTTP 服务；
- 每次搜索拿 top-3 文档。

也就是说，Search-R1 的“搜索能力”不是内嵌在 trainer 里的，而是通过配置接入的。

---

## 3. `train_grpo.sh` 和它到底差在哪

再看：`train_grpo.sh:1-82`。

你会发现它的整体结构几乎一样，但有几个关键不同点。

### 3.1 最本质的差别：advantage 估计方式不同

最关键这一行：

- `train_grpo.sh:41`

```bash
algorithm.adv_estimator=grpo
```

这意味着后面不会走 GAE，而会走 GRPO 的组内相对比较逻辑。

### 3.2 KL 处理方式不同

GRPO 脚本里会设置：

- `train_grpo.sh:47-60`

```bash
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

这和 PPO 路线不同。PPO 这边更常见的是在 reward 侧加入 KL penalty；GRPO 这边则更倾向于让 actor 直接使用 KL loss。

### 3.3 同一问题采样多条轨迹

- `train_grpo.sh:62`

```bash
actor_rollout_ref.rollout.n_agent=5
```

这一行是理解 GRPO 的关键。它意味着：

> 同一个问题会采样多条 agent 轨迹，然后在组内做相对比较。

这也是为什么后面你会看到 `index` / `uid` 这种字段被专门拿来分组。

不过这里还有一个很值得你提前知道的实现细节：

> 当前 `train_grpo.sh` 文件前面只定义了 `DATA_DIR`，但真正命令里引用的是 `$TRAIN_DATA_DIR` 和 `$TEST_DATA_DIR`（`train_grpo.sh:30-31`）。

所以如果你准备亲自跑 GRPO，最好先修正这两个变量，或者显式导出它们；否则脚本可能在读 parquet 路径时直接出错。

---

## 4. 这些 shell 参数会被谁接住

这里最容易卡住的点是：脚本里写了这么多 `a.b.c=x`，它们到底被谁解析？

答案是：**Hydra**。

入口在：`verl/trainer/main_ppo.py:104-105`

```python
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
```

这说明：

1. 默认配置来自 `verl/trainer/config/ppo_trainer.yaml`；
2. shell 里的 `xxx=yyy` 会作为 override 注入进去；
3. Python 里最终拿到的是“默认配置 + shell 覆盖之后”的结果。

所以你以后调试时，千万不要只看 yaml，也不要只看 shell。

> 真正生效的是两者合并后的 `config`。

这也是为什么在第 4 章我建议你断在 `main_task(config)` 开头，直接打印配置。

---

## 5. `ppo_trainer.yaml` 提供了哪些默认骨架

配置文件在：`verl/trainer/config/ppo_trainer.yaml:1-180`。

这个 yaml 很长，但你现在不用逐项背，只要先知道它提供了 5 组核心配置：

### 5.1 `data`
决定：
- 训练/验证文件位置；
- prompt 长度；
- response 长度；
- observation 长度；
- batch size。

例如：
- `ppo_trainer.yaml:1-17`

### 5.2 `actor_rollout_ref`
决定：
- actor 模型；
- rollout 后端；
- ref policy；
- 采样参数；
- FSDP / vLLM 行为。

例如：
- `ppo_trainer.yaml:18-92`

### 5.3 `critic`
决定：
- critic 模型；
- critic 学习率；
- value loss 相关参数。

例如：
- `ppo_trainer.yaml:93-127`

### 5.4 `retriever`
决定：
- 搜索服务 URL；
- topk。

例如：
- `ppo_trainer.yaml:148-150`

### 5.5 `algorithm`
决定：
- `adv_estimator` 是 `gae` 还是 `grpo`；
- `gamma`、`lam`；
- KL 策略；
- state masking 配置。

例如：
- `ppo_trainer.yaml:152-163`

所以可以把 yaml 理解成“完整骨架”，把 shell 理解成“本次实验的特化版本”。

---

## 6. `main_ppo.py` 的定位：它不是训练主战场，而是装配中心

文件：`verl/trainer/main_ppo.py:1-202`

第一次看这个文件时，很容易误以为它是“训练算法主逻辑”。其实不是。

更准确地说，它做的是：

> 把 tokenizer、reward、worker、resource pool 和 trainer 组装起来，然后把真正训练主循环交给 `RayPPOTrainer`。

所以这个文件应该用“装配视角”看，而不是“算法视角”看。

---

## 7. `main_ppo.py` 里最关键的三件事

### 7.1 选择 reward score function

先看：`main_ppo.py:25-29`

```python
def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
```

这说明当前项目的默认 reward 逻辑是：

- 根据 `data_source` 选择评分函数；
- 对 QA 任务，默认走 `qa_em.compute_score_em`。

这也是为什么 Search-R1 可以不用复杂 reward model，也能先跑起来：

- 它默认主要依赖 rule-based outcome reward。

### 7.2 定义 `RewardManager`

最值得看的位置是：`main_ppo.py:32-97`

这个类会做这些事：

1. 从 batch 里拿出 prompt 和 response；
2. 解码成完整字符串；
3. 读取 ground truth；
4. 按 `data_source` 选择评分函数；
5. 把 reward 写到 response 的最后一个有效 token 上。

这里你要特别记住最后一点：

> reward 不是平均铺在所有 token 上，而是先作为 outcome score 落到最后一个有效位置，再进入后续 advantage 计算。

这是后面理解 PPO/GRPO 非常关键的一步。

### 7.3 构造 trainer

看：`main_ppo.py:149-198`

这里主要做了三层装配：

- 定义 `role_worker_mapping`
- 定义 `resource_pool_spec`
- 构造 `RayPPOTrainer`

尤其是这段：

```python
trainer = RayPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    ray_worker_group_cls=ray_worker_group_cls,
    reward_fn=reward_fn,
    val_reward_fn=val_reward_fn,
)
```

这就是把“配置、数据打分逻辑、分布式 worker、tokenizer”全部交给训练器的时刻。

---

## 8. `main_task(config)` 里最值得你调试的变量

如果你以后用 `pdb` 断在 `main_task(config)`，我建议优先看这些变量：

```python
p config.actor_rollout_ref.model.path
p config.data.train_files
p config.data.val_files
p config.retriever.url
p config.algorithm.adv_estimator
p config.max_turns
```

为什么？

因为它们分别回答了 6 个最关键的问题：

1. 用什么基座模型？
2. 训练数据在哪里？
3. 验证数据在哪里？
4. 搜索服务地址是什么？
5. 你现在到底走 PPO 还是 GRPO？
6. 最多允许几轮搜索？

如果这几个变量都和你预期一致，后面大概率主线没走偏。

---

## 9. 这一章里你真正要建立的心智模型

我建议把这一章压缩成下面这张图：

```text
shell 脚本
  = 指定本次实验的参数

yaml 配置
  = 提供训练系统的默认骨架

main_ppo.py
  = 把 tokenizer / reward / workers / trainer 装起来

RayPPOTrainer
  = 真正执行 rollout、reward、advantage、update
```

如果你抓住这四层分工，后面读代码会顺很多。

---

## 10. 一个容易误解但非常重要的点

很多人第一次看这类项目时，会以为：

> “只要懂 `main_ppo.py`，就懂训练主逻辑了。”

其实不对。

更准确地说：

- `main_ppo.py` 负责“把舞台搭好”；
- `ray_trainer.py` 才负责“演员真正开始演”。

所以这一章的任务不是把所有训练细节看完，而是先回答：

- 入口在哪里；
- 配置怎么流动；
- PPO 和 GRPO 是怎么在入口层面分叉的；
- reward 和 worker 是怎样被装配进去的。

下一章，我们就会正式进入这个“真正开始演”的文件：`verl/trainer/ppo/ray_trainer.py`。

---

## 11. 本章小结

这一章最关键的结论有四个：

1. **PPO 和 GRPO 共用同一个 Python 训练入口。**
2. **差别主要体现在 shell 对 Hydra 配置的覆盖上。**
3. **`main_ppo.py` 的本质是装配中心，不是训练主战场。**
4. **真正的训练主循环在 `RayPPOTrainer.fit()` 里。**

如果你已经理解了这四点，下一章就可以顺理成章进入 Search-R1 的训练核心。