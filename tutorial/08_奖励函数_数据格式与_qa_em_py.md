# 第 08 章：数据格式、奖励函数与 `qa_em.py`

前几章我们已经把训练主线和 agent rollout 主线串起来了。

这一章要回答两个非常基础、但实际上极其关键的问题：

1. 模型训练时到底看到了什么样的数据？
2. 系统到底是怎样给一条轨迹打分的？

如果这两个问题不清楚，那么你后面看 PPO / GRPO 时就会一直悬着：

- reward 到底是从哪里来的？
- 它是给 token 打分，还是给整条回答打分？
- 数据里的 `index`、`reward_model`、`data_source` 到底有什么用？

这一章就是把这些地基讲稳。

---

## 1. 先记住一个总图

Search-R1 的训练数据流可以先概括成下面这样：

```text
原始问答样本
  -> 数据处理脚本生成统一格式
  -> 写成 parquet
  -> RLHFDataset 读取 parquet
  -> tokenizer/chat template 编码 prompt
  -> rollout 生成 response
  -> RewardManager 解码完整序列
  -> qa_em.py 从 <answer> 中抽答案并打分
```

也就是说：

- **数据格式**决定模型一开始看见什么；
- **奖励函数**决定模型最后学成什么样。

---

## 2. Search-R1 的训练样本长什么样

README 其实已经给了一个模板：
- `README.md:157-181`

形式大概是：

```python
data = {
    "data_source": data_source,
    "prompt": [{
        "role": "user",
        "content": question,
    }],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": solution
    },
    "extra_info": {
        "split": split,
        "index": idx,
    }
}
```

你可以把这个结构按语义拆成 4 部分。

### 2.1 `prompt`
这是模型真正看到的用户输入。

注意这里不是单纯字符串，而是 chat 结构：

```python
[{"role": "user", "content": question}]
```

这使得后面可以：
- 走 tokenizer 的 chat template；
- 兼容 instruction/chat 模型。

### 2.2 `reward_model.ground_truth`
这不是 reward model 网络，而是：

- 用来给 rule-based reward 提供标准答案。

在 NQ 这类任务里，它通常是：

```python
{"target": golden_answers}
```

### 2.3 `data_source`
它表示当前样本来自哪个任务，例如：
- `nq`
- `hotpotqa`
- `musique`

这个字段会决定：
- 该走哪种评分函数；
- 最后 validation 指标如何按数据集分组统计。

### 2.4 `extra_info.index`
这个字段先记住一个直觉就够：

> 它是数据侧给每个原始问题预留的分组标识。

后面到了 trainer 里，它会被复制成 `uid`，再真正用于 GRPO 的组内相对比较。这里先记字段语义，具体流转放到后文再展开。

---

## 3. `nq_search.py` 是怎么构造这些样本的

对应文件：
- `scripts/data_process/nq_search.py:42-101`

这个脚本主要做了三件事。

### 3.1 从原始数据集读取样本

它会加载：

```python
dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
```

也就是说，NQ 训练数据不是仓库里自带的，而是从 Hugging Face 数据集读取后再加工。

### 3.2 给 question 套统一 prompt 模板

最关键的位置是：
- `nq_search.py:27-39`

这里会构造一个带协议的前缀，例如：

- 要先在 `<think>` 中思考；
- 如果知识不足就用 `<search>`；
- 最终答案放进 `<answer>`。

这一步非常重要，因为它说明：

> Search-R1 的训练不是只给“裸问题”，而是给“带工具使用协议的问题”。

### 3.3 写成 parquet

最后：

- `nq_search.py:95-96`

会输出：
- `train.parquet`
- `test.parquet`

这也是后面 `train_ppo.sh` / `train_grpo.sh` 直接读取的文件。

---

## 4. `RLHFDataset` 到底做了什么

文件：
- `verl/utils/dataset/rl_dataset.py:58-155`

这个类的角色可以概括成一句话：

> 把 parquet 里的结构化样本，变成 trainer 可以直接使用的 prompt 张量 + 元信息。

### 4.1 先读 parquet

- `rl_dataset.py:96-103`

它会把一个或多个 parquet 文件拼成 dataframe。

### 4.2 再取出 chat prompt

- `rl_dataset.py:124-132`

```python
chat = row_dict.pop(self.prompt_key)
if self.tokenizer.chat_template:
    prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
else:
    prompt_with_chat_template = chat[0]['content']
```

这一步说明：

- 如果 tokenizer 自带 chat template，就走模板；
- 否则直接退化成原始文本内容。

所以 Search-R1 的数据加载逻辑是兼容不同模型风格的。

### 4.3 再 tokenize 成训练张量

- `rl_dataset.py:134-145`

它会生成：
- `input_ids`
- `attention_mask`
- `position_ids`

这三者就是后面 rollout 的起始输入。

### 4.4 最后把 `index` 单独取出来

- `rl_dataset.py:151-153`

```python
index = row_dict.get("extra_info", {}).get("index", 0)
row_dict["index"] = index
```

这一步很有 Search-R1 味道，因为它相当于提前为后面的 GRPO 分组埋好了钩子。

---

## 5. 一条样本真正进入训练时，长什么样

当样本经过 `RLHFDataset` 之后，它大概可以分成两层：

### 5.1 张量部分
例如：
- `input_ids`
- `attention_mask`
- `position_ids`

这些是模型生成需要的。

### 5.2 非张量部分
例如：
- `data_source`
- `reward_model`
- `ability`
- `extra_info`
- `index`

这些是：
- reward 计算需要的；
- 日志统计需要的；
- GRPO 分组需要的。

这一点很重要，因为 Search-R1 训练时不是只在搬运 token 张量，而是同时在携带“语义标签”和“监督信息”。

---

## 6. Search-R1 的 reward 设计，到底简单在哪里

现在进入奖励函数部分。

默认 reward 逻辑主要在两个地方：

- `verl/trainer/main_ppo.py:25-97`
- `verl/utils/reward_score/qa_em.py:1-138`

你会发现一个很重要的事实：

> 默认 reward 并不复杂，它主要是结果型规则奖励（rule-based outcome reward）。

也就是说，它不会去逐步给中间推理打密集奖励，而是主要看：

- 最终答案对不对；
- 格式是否合规。

---

## 7. `RewardManager` 在干什么

文件：
- `verl/trainer/main_ppo.py:32-97`

这个类是 reward 侧的总入口。

它主要做 5 步。

### 第一步：从 batch 里取出 prompt 和 response

它会根据 `attention_mask` 识别：
- prompt 的有效长度；
- response 的有效长度。

### 第二步：把 prompt 和 response 拼回完整序列

```python
sequences = torch.cat((valid_prompt_ids, valid_response_ids))
sequences_str = self.tokenizer.decode(sequences)
```

为什么要这么做？

因为 `qa_em.py` 需要看到完整字符串，才能从中抽出 `<answer>...</answer>`。

### 第三步：取出 ground truth

```python
ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
```

也就是说，标准答案不是从外部另查的，而是样本自带的。

### 第四步：根据 `data_source` 选评分函数

默认 QA 数据会走：
- `qa_em.compute_score_em`

### 第五步：把 score 放到最后一个有效 response token 上

```python
reward_tensor[i, valid_response_length - 1] = score
```

这是非常关键的一点。

> outcome reward 先只落在最后一个有效 token 上，而不是一开始就平均分配到整段 response。

这样做的结果是：
- reward 保持“最终结果导向”的语义；
- 后面的 GAE / GRPO 再决定如何传播这个信号。

---

## 8. `qa_em.py` 到底怎样打分

文件：
- `verl/utils/reward_score/qa_em.py:1-138`

这个文件里最值得你重点看的是三个函数：

### 8.1 `extract_solution()`
位置：`qa_em.py:62-82`

它会从完整字符串中找：

```python
answer_pattern = r'<answer>(.*?)</answer>'
```

但这里有一个**必须明确提醒**的实现细节：当前代码不是“只要找到 `<answer>` 就取最后一个”，而是：

- 如果匹配数量小于等于 1，直接返回 `None`；
- 只有匹配数量大于等于 2，才返回最后一个 `<answer>...</answer>` 中的内容。

也就是说，从当前实现看：

> 只有一个 `<answer>` 的正常回答，也可能因为这段逻辑而被判成“没有抽到答案”。

这不是一般意义上最自然的写法，但它确实是当前代码行为，所以你以后调试 reward 时一定要盯住这个点。

### 8.2 `em_check()`
位置：`qa_em.py:36-46`

它会把预测答案和 gold answer 都做 normalize，再做 exact match。

### 8.3 `compute_score_em()`
位置：`qa_em.py:85-110`

整体逻辑是：

1. 先抽答案；
2. 如果根本抽不到 `<answer>`，直接给 0；
3. 如果抽到了，再看和 `ground_truth['target']` 是否 exact match；
4. 命中则给 `score`，否则给 `format_score`。

### 8.4 一个特别值得注意的细节

- `qa_em.py:77-82`

如果 `<answer>` 匹配数量小于等于 1，它会返回 `None`。

这意味着当前 reward 实现存在一个很反直觉的现象：

- 不是“抽不到 `<answer>` 才给 0”；
- 而是“即使只有一个 `<answer>`，也可能被当成没抽到答案”。

所以更准确的说法是：

> 当前代码对 `<answer>` 的抽取逻辑带有一个额外假设，调试时不能想当然地把“出现了一个 `<answer>`”等价于“reward 一定能正常抽到答案”。

这里你以后如果遇到“看起来答案明明对了但 reward 还是 0”的情况，首先就该来检查这一层。

---

## 9. 为什么说 Search-R1 的 reward 主要是“结果导向”

因为从实现上看：

- reward 几乎不关心中间 `<think>` 是否优雅；
- 不关心 query 是否“看起来漂亮”；
- 主要关心最终 `<answer>` 能不能命中标准答案。

这和论文思路是一致的：

> 通过 outcome reward，让模型自己摸索怎样搜索、怎样推理，最终把答案做对。

对你这种刚入门 RL 的读者来说，这里其实很好理解：

- 奖励不是教你每一步怎么走；
- 奖励只是告诉你最后有没有走对。

---

## 10. 这种 reward 设计的优点和限制

### 10.1 优点

#### 优点一：简单、便宜、可扩展
不需要人工标中间推理轨迹。

#### 优点二：和搜索类任务很契合
因为很多题目的正确搜索路径不唯一。

#### 优点三：容易跨数据集复用
只要你能准备：
- prompt
- ground_truth

通常就能接进来。

### 10.2 限制

#### 限制一：credit assignment 更难
因为 reward 很晚才出现。

#### 限制二：中间行为质量不一定直接受约束
模型可能学会一些“能碰巧答对”的策略，而不一定是最优搜索策略。

#### 限制三：高度依赖答案抽取协议
如果 `<answer>` 标签格式错了，可能直接拿不到分。

这也是为什么格式协议在 Search-R1 里如此重要。

---

## 11. 从数据和 reward 的角度再看 `index`

现在你应该更容易理解 `index` 为什么重要了。

它不仅是数据字段，还在后面承担了一个 RL 语义：

- “这些 rollout 属于同一个原始问题”。

不过更精确地说，数据集先提供的是 `index`；到了 trainer 里，它会被复制成 `uid`，后续 GRPO 真正用来分组比较的是 `uid`。

在 PPO/GAE 里，这个字段的重要性还没那么突出；
但在 GRPO 里，它几乎是组内比较成立的前提。

所以你可以把 `index` 理解成：

> 从数据处理阶段就提前埋好的“组结构标签”，后面会在 trainer 中转成真正参与分组计算的 `uid`。

---

## 12. 这一章你最该盯住的变量

如果你想通过调试理解这一章，我建议重点看：

### 数据侧
```python
p row_dict.keys()
p row_dict['prompt']
p row_dict['reward_model']
p row_dict['index']
```

### reward 侧
```python
p sequences_str
p ground_truth
p score
p reward_tensor[i]
```

### `qa_em.py` 侧
```python
p answer
p ground_truth['target']
p normalize_answer(answer)
```

这些变量能帮你把“数据格式”和“reward 信号”真正连起来。

---

## 13. 本章小结

这一章最关键的结论有五个：

1. **Search-R1 的训练样本是结构化样本，不只是字符串。**
2. **`RLHFDataset` 负责把 parquet 样本变成 prompt 张量和元信息。**
3. **默认 reward 以 rule-based outcome reward 为主。**
4. **`RewardManager` 会把最终 score 放到最后一个有效 response token 上。**
5. **`index` 从数据处理阶段开始，就在为 GRPO 的分组比较服务。**

下一章，我们就把 PPO、GAE 和 GRPO 的算法实现细节讲透。