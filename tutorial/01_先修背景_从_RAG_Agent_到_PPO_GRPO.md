# 第 01 章：先修背景——从 RAG、Agent 到 PPO/GRPO

这一章不是在讲 Search-R1 的具体实现，而是在帮你补齐后面读代码一定会用到的背景。

你已经熟悉深度学习和 PyTorch，所以这里我不会从最基础的神经网络讲起；但因为你对 RL 和 LLM 训练机制还不够熟，所以我会把重点放在：

- Search-R1 想解决什么问题；
- 它和普通 RAG 有什么本质区别；
- 为什么这里会出现 PPO、GRPO 这类 RL 算法；
- 这些概念在代码里分别对应什么模块。

---

## 1. 一句话理解 Search-R1

Search-R1 可以概括成一句话：

> 它想训练出一种 LLM：遇到不会的问题时，不是硬猜，而是能在推理过程中主动搜索、读回结果、继续推理，最后再给答案。

注意这里的关键词是：

- **训练出**，不是只靠 prompt 临时凑；
- **推理过程中主动搜索**，不是固定先检索再生成；
- **继续推理**，说明搜索不是终点，而是推理链中的一个动作。

这也是论文《Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning》的核心目标。

---

## 2. 它和普通 RAG 的区别是什么

这是理解 Search-R1 的第一关键点。

### 2.1 普通 RAG 的典型流程

普通 RAG 往往是这样：

1. 用户提出问题；
2. 系统先拿这个问题去检索；
3. 把检索到的文档拼进 prompt；
4. 模型基于这些材料生成答案。

可以把它记成：

```text
问题 -> 检索 -> 拼接上下文 -> 生成答案
```

这种做法很有效，但它有一个明显限制：

> 检索通常发生在“回答之前”，而不是“思考过程中”。

也就是说，模型没有真的学会：

- 什么时候该搜；
- 要搜什么关键词；
- 搜到第一批材料后，下一步是否还要继续搜；
- 如何根据中间结果改写下一次查询。

### 2.2 Search-R1 的流程

Search-R1 更像下面这样：

```text
问题 -> 先想一想 -> 如果知识不够就发起搜索 -> 读回搜索结果 -> 继续想 -> 必要时再搜 -> 最后回答
```

也可以简化成：

```text
reason -> search -> observe -> reason -> answer
```

这就意味着：

- 搜索不再是固定预处理步骤；
- 搜索成为模型在生成过程中的一个**可学习动作**；
- 整个过程更像一个 agent 在和外部环境交互。

所以 Search-R1 与其说是“RAG 的一个小改版”，不如说是：

> **把搜索工具纳入到推理轨迹中，再用强化学习去优化这条轨迹。**

---

## 3. 为什么这里会出现 Agent 视角

如果你用 RL 的语言来描述 Search-R1，会发现它非常自然。

### 3.1 可以把它看成一个序列决策问题

在这个项目里，你可以粗略地这样对应：

- **state（状态）**：当前 prompt、已经生成的推理内容、已经拿到的搜索结果；
- **action（动作）**：继续输出推理文本、发起 `<search>...</search>`、或者给出 `<answer>...</answer>`；
- **observation（观察）**：搜索引擎返回的结果，会被包进 `<information>...</information>`；
- **reward（奖励）**：最终答案对不对；
- **trajectory（轨迹）**：从问题开始，到若干轮“推理/搜索/观察”，再到最后回答的完整过程。

这就是典型的 agent 视角。

### 3.2 为什么这个视角很重要

因为一旦你接受“搜索是 action”这件事，很多代码就容易理解了：

- 为什么会有多轮循环；
- 为什么要统计 action 合法性；
- 为什么生成结果里既有 response，也有 observation；
- 为什么 reward 不只是给语言模型打分，而是在给一条交互轨迹打分。

这也是 `search_r1/llm_agent/generation.py` 为什么是全仓最关键文件之一：它实现的不是普通文本生成，而是**agent rollout**。

---

## 4. 为什么要用强化学习，而不是只做监督微调

这是第二个关键问题。

### 4.1 只做 SFT 的局限

如果你只做监督微调，往往需要大量“标准轨迹”数据，例如：

- 第一步应该怎么想；
- 第一步应该搜什么；
- 搜完之后第二步应该怎么想；
- 什么时候结束；
- 最终答案是什么。

这种数据非常贵，而且很难保证轨迹唯一正确。

对搜索类问题尤其如此：

- 两个人可能会搜不同关键词；
- 两条不同轨迹都可能得出正确答案；
- 中间步骤很难完全人工标注。

### 4.2 强化学习更适合“结果导向”的任务

RL 的一个优势是：

> 你不必规定模型每一步都怎么做，只需要告诉它最后做得好不好。

在 Search-R1 里，作者大量使用的是**结果型奖励**：

- 最终答对了，奖励高；
- 最终答错了，奖励低；
- 中间是否搜索、搜索几次、搜索词长什么样，由模型自己逐渐学出来。

这就很符合这类任务的性质。

你可以把它类比成：

- SFT 更像老师手把手教每一步；
- RL 更像老师只看最后是否解对题，但允许你自己摸索过程。

---

## 5. PPO 是什么，先抓住直觉就够了

你现在不需要一口气掌握完整公式，先理解直觉更重要。

### 5.1 策略梯度在这里是什么意思

在 RL 里，策略就是：

> 给定当前状态，模型应该采取什么动作的概率分布。

放到 Search-R1 中，策略其实就是：

- 当前我已经看到这些上下文；
- 下一步我是继续想、发起搜索，还是直接回答；
- 如果要搜索，我应该生成什么 query；
- 如果要回答，我应该输出什么答案。

这本质上就是一个条件概率模型，也就是 LLM 本来就在做的事情。

所以用策略梯度训练 LLM，并不是完全陌生的东西；只是现在优化目标不再只是“拟合下一个 token”，而是“让整条轨迹更可能拿到高奖励”。

### 5.2 PPO 解决什么问题

如果你直接根据奖励大幅更新策略，训练很容易崩。

PPO（Proximal Policy Optimization）的核心思想很朴素：

> 更新可以，但别一步走太猛；新策略不要离旧策略太远。

它通常会做几件事：

1. 先用当前策略采样一批轨迹；
2. 根据奖励估计哪些动作更好；
3. 更新策略，但通过 clip 或 KL 约束控制步子大小。

在这个仓库里，你会在 `verl/trainer/ppo/core_algos.py` 里看到这些对应实现：

- `compute_policy_loss(...)`
- `compute_value_loss(...)`
- `kl_penalty(...)`
- `compute_gae_advantage_return(...)`

---

## 6. GAE 是什么，为什么 PPO 常和它一起出现

如果最终奖励只在回答结束时才出现，那么前面很多 token、很多动作都像“延迟记账”。

GAE（Generalized Advantage Estimation）是一个很常见的办法，用来估计：

> 某一步动作，相比“正常水平”到底好多少。

你可以先把它理解成一种更平滑的 credit assignment 方法：

- 不是简单把最终奖励粗暴地分给所有步骤；
- 而是结合 value 估计，给每一步算 advantage。

这就是为什么在 PPO 路线里通常还需要一个 **critic/value model**：

- actor 负责生成动作；
- critic 负责估计某个状态值多少钱。

在这个仓库中：

- PPO 路线使用 `algorithm.adv_estimator=gae`；
- 同时会启动 critic；
- 对应训练脚本是 `train_ppo.sh`。

---

## 7. GRPO 又是什么，为什么它看起来更“省事”

GRPO 可以先用一句话理解：

> 它不强依赖单独的 critic，而是让同一个问题的多条回答互相比较，做组内相对学习。

在 Search-R1 里，这个思路非常直观：

- 对同一个问题，采样多条轨迹；
- 有的轨迹搜得更好、答案更准；
- 有的轨迹虽然格式对，但搜得没价值，或者答案不对；
- 用组内相对得分去构造 advantage。

对应代码上你会看到：

- `train_grpo.sh` 里设置了 `algorithm.adv_estimator=grpo`；
- 同时 `actor_rollout_ref.rollout.n_agent=5`，表示同一问题会采样多条轨迹；
- `verl/trainer/ppo/core_algos.py` 里有 `compute_grpo_outcome_advantage(...)`；
- `verl/trainer/ppo/ray_trainer.py` 会先把样本 `index` 复制成 `uid`，真正把 `uid` 作为 GRPO 分组依据。

这对你来说只要先记住一个直觉：

> PPO 更像“actor + critic”路线；GRPO 更像“同题多样本相对比较”路线。

后面讲到代码时，我们再详细展开。

---

## 8. Search-R1 在代码里如何对应这些概念

把前面的概念和代码先对齐一下。

### 8.1 搜索动作的最小闭环

- `infer.py`

这是最容易读懂的文件。它直接把下面这件事跑通：

```text
模型生成 -> 如果发现 <search>query</search> -> 调本地检索服务 -> 把结果包成 <information>...</information> -> 继续生成
```

它很像一个“教学版 Search-R1”。

### 8.2 多轮搜索 agent 的正式实现

- `search_r1/llm_agent/generation.py`

这是训练时真正使用的多轮 agent loop。它负责：

- 截断输出到 `</search>` 或 `</answer>`；
- 解析当前动作；
- 调搜索服务；
- 把 observation 回填进上下文；
- 多轮推进直到结束。

### 8.3 RL 训练入口

- `verl/trainer/main_ppo.py`

这个文件负责：

- 读配置；
- 初始化 tokenizer 和 Ray worker；
- 构造 reward manager；
- 启动 `RayPPOTrainer`。

### 8.4 训练主循环

- `verl/trainer/ppo/ray_trainer.py`

这是全仓最像“总调度中心”的地方。它负责：

- 加载 batch；
- rollout 生成轨迹；
- 调 reward；
- 算 advantage；
- 更新 actor / critic。

### 8.5 PPO / GRPO 算法细节

- `verl/trainer/ppo/core_algos.py`

这个文件更偏“算法核心”，包括：

- GAE advantage；
- GRPO outcome advantage；
- PPO 的 policy loss / value loss；
- KL penalty。

---

## 9. 你现在应该先形成的 mental model

在继续读代码之前，我建议你先牢牢记住下面这个模型：

```text
Search-R1 = 会推理的 LLM + 可调用的搜索工具 + 多轮 agent 交互循环 + 基于最终答案的 RL 优化
```

再拆开一点：

```text
输入问题
  -> LLM 先生成推理/动作
  -> 如果动作是 search，就访问检索服务
  -> 把检索结果作为 observation 放回上下文
  -> 继续生成
  -> 最终输出 answer
  -> 根据答案正确性给 reward
  -> 用 PPO / GRPO 更新策略
```

如果你理解了这张图，后面所有代码几乎都只是这张图的不同实现层次。

---

## 10. 本章小结

这一章最重要的不是记定义，而是建立三个判断：

1. **Search-R1 不是普通 RAG。**
   因为它不是“先检索再生成”，而是“推理过程中按需搜索”。

2. **搜索在这里是 action。**
   所以它天然适合用 agent / RL 的视角理解。

3. **PPO / GRPO 不是配角。**
   它们决定了模型如何从“答题结果”反过来学习“什么时候搜、搜什么、搜完怎么接着想”。

---

## 11. 推荐额外读物

如果你想先补背景，再继续读后面的章节，可以看这些资料：

### 强化学习 / PPO
- PPO 中文概览（适合先建立直觉）：<https://www.cnblogs.com/BlogNetSpace/p/18806099>
- PPO 中文长文（更细一点）：<https://zhuanlan.zhihu.com/p/614115887>

### RAG / 检索增强生成
- LangChain4j 的 RAG 中文教程：<https://docs.langchain4j.info/tutorials/rag>
- Prompt Engineering Guide 的 RAG 中文说明：<https://www.promptingguide.ai/zh/techniques/rag>
- 一篇中文 RAG 概览：<https://zhuanlan.zhihu.com/p/675509396>

### Search-R1 论文
- 论文原文：<https://arxiv.org/pdf/2503.09516>

如果你继续往下读，我建议下一章直接进入仓库全景图。