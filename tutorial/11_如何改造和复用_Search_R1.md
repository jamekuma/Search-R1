# 第 11 章：如何改造和复用 Search-R1

到这一章，我们已经把 Search-R1 的核心主线基本讲完了：

- 为什么它不是普通 RAG；
- 最小闭环如何工作；
- 训练入口如何启动；
- PPO/GRPO 如何接上；
- 搜索服务如何提供外部知识。

这一章就进入一个更实际的问题：

> 如果你想把 Search-R1 迁移到自己的任务，应该从哪里改起？哪些地方是任务相关的，哪些地方是框架相关的？

这一章会尽量站在“算法工程师要动手改代码”的视角来讲，而不是泛泛给建议。

---

## 1. 先给一个总原则

如果你以后要改 Search-R1，我建议你先记住一个原则：

> **优先改数据、reward 和搜索接口；谨慎改 trainer 主流程；更谨慎改底层分布式 worker。**

为什么？

因为对大多数任务迁移来说：

- 任务差异主要体现在样本长什么样；
- reward 怎么定义；
- 搜索用什么工具；

而不是：
- PPO 主循环要不要重写；
- Ray worker 要不要推倒重来。

也就是说，Search-R1 最适合你改的地方，是“任务层”和“工具层”；不是一上来就动“训练基础设施层”。

---

## 2. 哪些模块是任务相关的

我建议把“任务相关”的地方归为下面 4 类。

### 2.1 数据处理脚本

例如：
- `scripts/data_process/nq_search.py`

这是最直接的任务相关模块。

如果你换成自己的任务，最先该改的通常就是这里：

- 原始样本从哪里读；
- prompt 如何构造；
- ground truth 如何组织；
- `index` 如何分配。

### 2.2 reward score function

例如：
- `verl/utils/reward_score/qa_em.py`
- `verl/trainer/main_ppo.py` 中的 `_select_rm_score_fn()`

如果你的任务不是 QA exact match，而是：
- 多选题；
- 数值预测；
- 结构化抽取；
- 带证据要求的回答；

那么 reward 一定要改。

### 2.3 prompt 协议

例如：
- `scripts/data_process/nq_search.py` 里的模板
- `infer.py` 里的 prompt

如果你希望模型：
- 使用不同工具；
- 输出不同标签；
- 采用不同终止格式；

这些协议必须保持一致。

### 2.4 搜索服务

例如：
- `search_r1/search/retrieval_server.py`
- `search_r1/search/serp_search_server.py`
- `search_r1/search/google_search_server.py`

如果你换成自己的知识库、搜索 API 或垂直搜索引擎，这一层通常也需要改。

---

## 3. 哪些模块更偏框架通用，不建议一开始乱改

### 3.1 `verl/trainer/ppo/ray_trainer.py`

这是训练主循环中枢。

除非你明确知道自己要改什么，否则不建议一开始就大改。因为这里牵涉：

- dataloader
- rollout
- reward
- KL
- advantage
- actor / critic update
- validation

改错一个地方，影响会扩散很大。

### 3.2 `verl/workers/*`

这里更多是分布式执行与模型更新的底层支撑。

如果你现在的目标是“把 Search-R1 迁移到别的问答/搜索任务”，通常完全不需要先动这里。

### 3.3 `verl/single_controller/*`

这是 Ray / worker group 层，离任务语义更远。

除非你是在研究训练框架本身，否则先把它当黑盒是更合理的。

---

## 4. 如果你想迁移到自己的 QA 数据，最小改法是什么

这是最常见的需求。

### 4.1 你至少需要改三件事

#### 第一：写自己的数据处理脚本
输出格式尽量保持和现有样本一致：

```python
{
    "data_source": ...,
    "prompt": [...],
    "reward_model": {
        "style": "rule",
        "ground_truth": ...,
    },
    "extra_info": {
        "split": ...,
        "index": ...,
    }
}
```

#### 第二：决定 reward 怎么打
如果还是 QA exact match，那你甚至可以复用：
- `qa_em.py`

如果不是，就在：
- `verl/utils/reward_score/`

下新增你自己的打分函数。

#### 第三：决定搜索语料和索引怎么来
如果你的知识是私有语料：
- 先准备 corpus jsonl；
- 再建 BM25 / dense index；
- 再起 retrieval server。

这三步通常就够了。

---

## 5. 如果你想迁移到“不是 QA exact match”的任务，该怎么想

这个问题比前一个更值得认真讲。

因为 Search-R1 现在的默认实现很强地绑定在：

- 最终 `<answer>`
- exact match / sub exact match

上。

如果你的任务换成：
- 需要生成一段解释；
- 需要抽结构化字段；
- 需要输出代码片段；
- 需要判断支持证据是否充分；

你就不能再只靠 `qa_em.py`。

### 5.1 这时最该改哪里

#### 入口一：`_select_rm_score_fn()`
在：
- `main_ppo.py:25-29`

这里决定不同 `data_source` 走哪套 reward。

#### 入口二：新增 reward 文件
例如放在：
- `verl/utils/reward_score/my_task.py`

#### 入口三：必要时改变输出协议
例如：
- 最终不是 `<answer>...</answer>`，而是别的结构。

但要注意：

> 一旦你改了输出协议，数据处理、infer、generation、reward 这些层最好一起保持一致。

否则系统会出现“模型按一种协议生成、reward 却按另一种协议解析”的错位。

---

## 6. 如果你想换工具，而不只是换语料

这是 Search-R1 很有潜力的一点。

当前项目默认工具是“搜索工具”，更具体地说，是一个通过文本协议触发、底层由检索服务承载的外部工具。

### 6.1 现有协议是什么

当前主要是：
- `<search>query</search>`
- 返回 `<information>...</information>`

### 6.2 如果你想换成别的工具

比如：
- 数据库查询；
- 代码执行；
- 论文检索；
- 企业内知识库查询。

本质上要改三层：

#### 第一层：动作协议
`generation.py` 目前只识别：
- `search`
- `answer`

如果你想加新动作，`postprocess_predictions()` 就得改。

#### 第二层：环境执行逻辑
`execute_predictions()` 目前只会处理：
- 搜索；
- 结束；
- 非法动作。

如果有新工具，这里要扩展成更多 action 分支。

#### 第三层：observation 格式
现在默认用 `<information>...</information>`。

如果你想引入更多工具，最好设计更清晰的 observation 协议，不然多工具场景会很乱。

---

## 7. 如果你只是想做一个“小改版 Search-R1”，最推荐的改法是什么

我给你一个很实用的优先级建议。

### 第一优先级：换数据 + 换语料
这是最安全也最常见的方式。

你改：
- 数据处理脚本；
- corpus / index；
- reward 的小部分。

你不改：
- trainer 主循环；
- worker 架构。

### 第二优先级：换 reward
适合：
- 任务还是同类 agent 搜索任务；
- 但评价标准和 QA-EM 不同。

### 第三优先级：换搜索服务
适合：
- 你已经有自己的内部检索系统；
- 或你要接入在线搜索 API。

### 最后才考虑：改 rollout 主体
例如改：
- 新动作空间；
- 多工具调用；
- tree-style search。

这一步最有研究价值，但也最容易把系统复杂度拉高。

---

## 8. 什么时候你需要改 `generation.py`

这是个很关键的问题。

### 不需要改 `generation.py` 的情况

如果你只是：
- 换一个新 QA 数据集；
- 换检索语料；
- 换 reward 形式；
- 换模型；

通常都不需要动 `generation.py`。

### 需要改 `generation.py` 的情况

如果你要改变 agent 行为本身，例如：
- 不再只支持 `search` / `answer` 两类动作；
- 想做多工具调用；
- 想让 observation 有不同类型；
- 想支持并行搜索、多 query 分解、路由等更复杂策略；

这时才真正应该动它。

也就是说：

> `generation.py` 更接近“agent 行为定义层”，不是普通任务定制层。

---

## 9. 一个常见误区：一开始就大改训练器

我非常建议你避免这个误区。

很多人看到 Search-R1 后，会马上想：

- 我要把 PPO 主循环重写一下；
- 我要顺手改一下 worker 设计；
- 我要把一堆逻辑挪来挪去。

但对大多数任务迁移来说，这种做法收益并不高，代价却很大。

更合理的思路是：

1. 先确认你只是“任务换了”，还是“agent 机制换了”；
2. 如果只是任务换了，优先改数据、reward、检索服务/检索器；
3. 只有 agent 机制真的变了，再去动 `generation.py` 甚至 trainer。

---

## 10. 一个适合你的复用路线图

结合你的背景，如果你未来想自己基于 Search-R1 做实验，我建议按这个顺序来：

### 路线 A：先做“同范式迁移”
比如：
- 仍然是问答；
- 仍然是 search agent；
- 但换成你自己的领域语料。

你要做的事：
- 写数据处理；
- 建索引；
- 调 reward；
- 跑 PPO/GRPO。

这是最适合第一步上手的。

### 路线 B：再做“reward 研究”
比如：
- 不只看最终答案对不对；
- 还考虑格式、证据、搜索效率等。

这一步很适合你以后进一步理解 RL 设计空间。

### 路线 C：最后做“多工具 agent 扩展”
比如：
- 搜索 + 数据库；
- 搜索 + 代码执行；
- 搜索 + 重排序 + 验证。

这一步就更偏研究和系统设计了。

---

## 11. 本章小结

这一章最关键的结论有六个：

1. **迁移 Search-R1 时，优先改数据、reward 和搜索接口。**
2. **大多数任务迁移不需要先动 trainer 主循环。**
3. **`generation.py` 属于 agent 行为定义层，只有在动作机制变化时才优先修改。**
4. **如果任务还是 QA 搜索型，很多基础设施都可以直接复用。**
5. **如果评价标准变化，reward 设计会成为第一修改点。**
6. **Search-R1 很适合作为“搜索型 RL agent”的研究底座，而不只是一个固定任务工程。**

到这里，你已经能判断：如果以后要基于 Search-R1 做自己的实验，应该优先改哪里，又有哪些地方最好先别动。

- 你知道它想解决什么问题；
- 知道仓库结构；
- 知道最小运行闭环；
- 知道怎么跑起来和调试；
- 知道训练主流程、agent 主循环、reward、PPO/GRPO、检索系统；
- 也知道以后该怎样基于它继续做自己的实验。