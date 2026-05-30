# 第 07 章：`generation.py`——Search-R1 最核心的 agent 循环

如果让我在整个仓库里只选一个最能体现 Search-R1 特点的文件，我会选：

- `search_r1/llm_agent/generation.py`

原因很简单：

> Search-R1 最核心的创新，不是把 PPO 公式换了一下，而是把 rollout 从“普通文本生成”改造成“可多轮搜索的 agent 交互循环”。

而这个交互循环的主体，就是这里实现的。

---

## 1. 为什么这一章值得单独讲

你前面已经看过：

- `infer.py` 展示了最小闭环；
- `ray_trainer.py` 展示了训练总流程。

但两者之间还差一层：

- `infer.py` 太简化；
- `ray_trainer.py` 太宏观。

`generation.py` 刚好处在中间：

- 比 `infer.py` 更正式、更适合批量训练；
- 比 `ray_trainer.py` 更聚焦，更能看到 Search-R1 的核心增量。

所以可以把它理解成：

> Search-R1 的“环境交互内核”。

---

## 2. 先看这个文件在做什么

这个文件的主角有两个：

### 2.1 `GenerationConfig`
位置：`generation.py:13-24`

它定义了多轮 agent rollout 需要的主要参数：

- `max_turns`
- `max_start_length`
- `max_prompt_length`
- `max_response_length`
- `max_obs_length`
- `num_gpus`
- `search_url`
- `topk`

这说明 Search-R1 的 rollout 不只是“生成多少 token”，还明确关心：

- 最多进行几轮搜索；
- 起始 prompt 留多长；
- observation 最长能有多长；
- 搜索服务地址是什么。

### 2.2 `LLMGenerationManager`
位置：`generation.py:25-469`

这是这一章真正要讲的核心类。

它的职责可以用一句话概括：

> 管理一批样本在“生成—搜索—观察—再生成”环境中的多轮 rollout。

---

## 3. 整个 `run_llm_loop()` 的大图

主函数在：
- `generation.py:220-319`

建议你先记住下面这张图：

```text
初始化 rolling state
  -> 对 active 样本生成一轮 response
  -> 截断到 </search> 或 </answer>
  -> 解析动作
  -> 如果是 search，则调检索服务
  -> 生成 next observation
  -> 把 response + observation 拼回上下文
  -> 更新 active_mask
  -> 重复直到 max_turns
  -> 最后再做一次最终生成
  -> 组装成 DataProto 输出
```

这张图就是 Search-R1 的 agent rollout 本体。

---

## 4. 为什么函数一开始要维护 left side / right side

看：`generation.py:223-224`

```python
original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
```

第一次看这里会很怪：为什么要分 left side / right side？

### 4.1 直觉理解

可以这样理解：

- **left side**：原始 prompt 的起始部分；
- **right side**：后续 rollout 过程中不断追加的内容。

后续追加内容既包括：
- 模型生成的 response；
- 也包括搜索返回的 observation。

所以 Search-R1 不是简单地“把当前 prompt 改长”，而是在内部很明确地区分：

- 哪些是起始输入；
- 哪些是 rollout 过程中新长出来的内容。

这对后面构造：

- `prompts`
- `responses`
- `info_mask`

都很重要。

---

## 5. `active_mask` 是整个循环的关键控制器

看：`generation.py:226-230`

```python
active_mask = torch.ones(...)
turns_stats = torch.ones(...)
valid_action_stats = torch.zeros(...)
valid_search_stats = torch.zeros(...)
```

这里其实是在做一件很 agent 的事：

> 同一个 batch 里的不同样本，可能在不同轮次结束，所以要追踪哪些样本还“活着”。

### 5.1 `active_mask` 的作用

- `True`：这个样本还没结束，下一轮还要继续 rollout；
- `False`：这个样本已经回答完，或者已经不再需要继续。

这也是为什么 Search-R1 的 rollout 比普通单轮生成更像“环境交互”：

- 每个样本不是只生成一次；
- 而是每轮根据动作结果决定是否继续。

---

## 6. 主循环每轮在干什么

看：`generation.py:233-277`

每一轮可以拆成 6 步。

### 第一步：先裁掉无效 padding

```python
rollings.batch = self.tensor_fn.cut_to_effective_len(...)
```

作用：
- 减少无效 token；
- 控制输入长度；
- 保持 rollout 更高效。

### 第二步：只对 active 样本生成

```python
rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
gen_output = self._generate_with_gpu_padding(rollings_active)
```

这表示已经结束的样本不会再继续生成。

### 第三步：后处理响应

```python
responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
```

这一步很关键，它会把输出截断到：

- `</search>`
- 或 `</answer>`

对应实现是：`generation.py:54-75`

这一步的本质是：

> 不让模型一口气无限生成，而是把生成切成一个个“动作回合”。

### 第四步：执行动作

```python
next_obs, dones, valid_action, is_search = self.execute_predictions(...)
```

这是最核心的一步。它相当于环境的 `step()`。

### 第五步：更新 active 状态

```python
curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
active_mask = active_mask * curr_active_mask
```

如果某个样本已经给出 `<answer>`，它后面就不会再参与下一轮。

### 第六步：把 observation 拼回去

```python
rollings = self._update_rolling_state(...)
original_right_side = self._update_right_side(...)
```

也就是说，一轮 response 不会凭空消失，而是会和检索返回一起，成为下一轮生成的上下文。

---

## 7. `_postprocess_responses()` 为什么这么重要

位置：`generation.py:54-75`

里面最关键的一段是：

```python
responses_str = [resp.split('</search>')[0] + '</search>'
         if '</search>' in resp 
         else resp.split('</answer>')[0] + '</answer>'
         if '</answer>' in resp 
         else resp
         for resp in responses_str]
```

这段代码的作用非常明确：

- 如果输出里出现了 `</search>`，本轮就截到这里；
- 否则如果出现了 `</answer>`，本轮就截到这里；
- 否则保留原样。

### 7.1 它对应了什么思想

这相当于把 LLM 生成切成“环境 step 级别”的片段。

如果没有这一步，模型可能会：
- 一边生成搜索标签；
- 一边继续写一大段；
- 让系统无法在正确时机插入 observation。

所以这一步本质上是在把“自由文本生成”变成“受控 agent 行为”。

---

## 8. `execute_predictions()` 才是真正的环境 step

位置：`generation.py:353-405`

这是整个文件最值得精读的函数之一。

它会把模型输出解释成三种情况：

### 8.1 情况一：`answer`

```python
if action == 'answer':
    next_obs.append('')
    dones.append(1)
```

含义：
- 这个样本已经给出最终答案；
- 环境结束；
- 不再继续下一轮。

### 8.2 情况二：`search`

```python
elif action == 'search':
    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
    dones.append(0)
```

含义：
- 这不是结束，而是一次合法动作；
- 环境会返回 observation；
- 下一轮继续。

### 8.3 情况三：非法动作

```python
else:
    next_obs.append('My previous action is invalid...')
    dones.append(0)
```

这一步非常有意思。

Search-R1 没有在这里直接把样本判死，而是：

> 给模型返回一段错误反馈，让它下一轮重试。

这和很多 agent 系统的设计是一致的：

- 非法动作不一定终止；
- 而是把错误信息作为环境反馈返回给模型。

这也是为什么后面会统计：

- `valid_action_stats`
- `valid_search_stats`

---

## 9. `postprocess_predictions()`：动作是怎么被解析出来的

位置：`generation.py:407-436`

核心正则是：

```python
pattern = r'<(search|answer)>(.*?)</\1>'
```

这说明 Search-R1 的动作空间在文本层面非常清楚：

- `search`
- `answer`

也就是说，虽然模型表面上还是在生成 token，但系统在语义上已经把其中一部分 token 当成：

- 环境动作声明。

这是理解 Search-R1 的一个关键点：

> 动作不是额外的 API schema，而是嵌在文本协议中的结构化片段。

---

## 10. `batch_search()`：Search-R1 是怎样真正调搜索引擎的

位置：`generation.py:438-458`

这里的逻辑非常直接：

```python
payload = {
    "queries": queries,
    "topk": self.config.topk,
    "return_scores": True
}
return requests.post(self.config.search_url, json=payload).json()
```

这说明三件事：

1. Search-R1 并不依赖某个固定检索器实现；
2. 它只依赖一个统一的 HTTP 协议；
3. 只要你的搜索引擎服务能满足这个接口，就能被接入。

这也是为什么后面改造项目时，搜索引擎其实是比较容易替换的一层。

---

## 11. observation 为什么要包装成 `<information>`

看：`generation.py:391`

```python
next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
```

这说明环境返回不是裸文本，而是显式带标签的 observation。

### 11.1 这样设计有什么好处

至少有三个好处：

#### 好处一：模型知道“这不是我自己刚才想出来的”
`<information>` 明确告诉模型：
- 这段内容来自外部工具；
- 不是内部推理文本。

#### 好处二：后面更容易做 state masking
因为环境内容被明确包起来了，所以后面可以用：

- `<information>`
- `</information>`

去识别 observation token 边界。

#### 好处三：便于调试
你在 prompt 里一眼就能看出：
- 哪一段是模型自己生成的；
- 哪一段是检索系统返回的。

---

## 12. `_update_rolling_state()`：为什么它是循环能继续的关键

位置：`generation.py:93-118`

这个函数负责把三部分拼起来：

- 旧上下文；
- 当前 response；
- 当前 next observation。

对应代码：

```python
new_input_ids = self.tensor_fn.concatenate_with_padding([
    rollings.batch['input_ids'],
    cur_responses,
    next_obs_ids
])
```

这一步的意义非常直接：

> 下一轮生成看到的上下文，必须包含这一轮的动作和环境反馈，否则 agent 就没有“记忆”。

这就是多轮 agent loop 的本质。

---

## 13. `responses_with_info_mask` 到底在干嘛

这是第一次读时最容易忽略，但后面又非常重要的一块。

相关位置：
- `generation.py:120-167`
- `generation.py:339-342`

这里会额外维护一个：

- `responses_with_info_mask`

它不是普通 response，而是把 observation 位置特殊处理过的一份版本。

### 13.1 它的目的

本质上是为了后面构造：

- `info_mask`

在 `_compose_final_output()` 里你会看到：

```python
final_output['info_mask'] = torch.cat([
    ...,
    self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
], dim=1)
```

这意味着：

- trainer 后面能知道哪些 token 属于环境 information；
- 进而支持 state masking。

### 13.2 为什么这对 Search-R1 特别重要

因为 Search-R1 的输入序列里混了两种东西：

1. 模型自己生成的 token；
2. 检索系统返回的 token。

如果你不把两者区分开，后面策略学习就会变得很别扭。

所以 `info_mask` 是一个非常“Search-R1 特有”的设计点。

---

## 14. 为什么还有 `_generate_with_gpu_padding()`

位置：`generation.py:169-218`

这个函数属于比较工程化的辅助函数。

它解决的问题是：

> 当 active 样本数不能整除 GPU 数时，如何保证多 GPU 生成不会因为 batch size 对不齐而出问题？

它的做法是：
- 如果 batch 大小不整除 GPU 数，就复制几条 padding 样本；
- 生成结束后再把这些 padding 结果裁掉。

这一步和 Search-R1 的核心算法关系不大，但和“能不能稳定跑起来”关系很大。

所以你可以把它看成一个很典型的：

- 算法主线之外，但工程上不可少的兼容层。

---

## 15. 最后一轮为什么要 `do_search=False`

看：`generation.py:278-310`

在主循环结束后，如果还有 active 样本，它会再做一次 final rollout：

```python
_, dones, valid_action, is_search = self.execute_predictions(
    responses_str, self.tokenizer.pad_token, active_mask, do_search=False
)
```

这一步很有意思。

### 15.1 它的直觉

前面 `max_turns` 轮已经允许模型多次搜索了。

最后这一轮更像是在说：

> 现在别再继续真的搜了，请把最后答案收束出来。

也就是说，它是在把多轮交互收尾成一个最终输出。

这也解释了 Search-R1 里的一个隐含设定：

- 搜索不是无限次的；
- 到达一定轮数后，系统会强制进入“收尾回答阶段”。

---

## 16. `meta_info` 为什么值得关注

看：`generation.py:312-319`

这里会记录很多 rollout 统计量：

- `turns_stats`
- `active_mask`
- `valid_action_stats`
- `valid_search_stats`

这些信息后面会进入 trainer 的 metrics 里。

这意味着 Search-R1 不只是关心最终 reward，还关心一些 agent 行为指标，例如：

- 平均执行了多少轮动作；
- 合法动作比例有多高；
- 合法搜索次数是多少；
- 有多少轨迹没有正常结束。

这些指标非常适合用来判断：

- 模型是不是学会了正确使用工具；
- 训练是不是在往合理方向发展。

---

## 17. 把 `infer.py` 和 `generation.py` 对起来看

为了帮助你建立更稳的直觉，我建议你把两者这样对照：

| `infer.py` | `generation.py` |
| --- | --- |
| 单问题 while 循环 | batch 化 `run_llm_loop()` |
| `get_query()` | `postprocess_predictions()` |
| `search()` | `batch_search()` |
| 拼 `<information>` | `execute_predictions()` 生成 next_obs |
| 手工维护 prompt | `_update_rolling_state()` |
| 无额外 mask | 维护 `info_mask` / `responses_with_info_mask` |

你会发现：

> `generation.py` 本质上就是 `infer.py` 的“训练版、批量版、可统计版、可与 PPO 数据流对接版”。

---

## 18. 本章最值得你记住的一句话

如果要把这一章压缩成一句话，我会这样说：

> `generation.py` 把 LLM 从“单轮生成器”变成了“在搜索环境中多轮行动的 agent”。

这正是 Search-R1 最核心的工程与算法增量。

---

## 19. 本章小结

这一章最关键的结论有六个：

1. **`run_llm_loop()` 是 Search-R1 的多轮 agent rollout 内核。**
2. **`execute_predictions()` 相当于环境的 `step()`。**
3. **`search` 和 `answer` 是通过文本协议表达的动作。**
4. **检索返回会被包装成 `<information>` observation，再拼回上下文。**
5. **`info_mask` 让 trainer 能区分“模型自己生成的 token”和“环境返回的 token”。**
6. **`generation.py` 是连接 `infer.py` 与 PPO 训练主流程的桥梁。**

下一章，我们会把训练样本和 reward 机制讲透：模型到底看到了什么数据，又是如何被打分的。