# 第 03 章：从 `infer.py` 看最小工作闭环

如果你只读一个文件来理解 Search-R1，我建议先读 `infer.py`。

原因很简单：

> 它用最少的工程复杂度，展示了 Search-R1 最核心的思想：模型一边生成，一边决定是否调用搜索，再把搜索结果喂回去继续生成。

这一章我们就围绕这个文件，先把“最小闭环”彻底讲明白。

---

## 1. 这个文件为什么这么重要

训练代码当然更完整，但训练代码里有很多会干扰理解的部分：

- Ray 分布式调度；
- actor / critic / ref policy；
- batch repeat、mask、padding；
- reward 和 advantage 计算。

而 `infer.py` 暂时不关心这些。

它只做一件事：

> 给定一个问题，让模型自己决定是否搜索；如果搜索，就请求检索服务；然后把结果加回上下文，继续生成，直到给出答案。

所以它特别适合建立第一性理解。

---

## 2. 文件的整体结构

你可以先把 `infer.py` 切成 5 段来看：

1. 定义问题与模型；
2. 写 prompt，告诉模型要用哪些标签；
3. 加载 tokenizer / model；
4. 定义停止条件与 query 提取函数；
5. 进入 `while True` 循环，反复执行“生成 -> 搜索 -> 回填”。

换成伪代码就是：

```python
prompt = build_prompt(question)
model, tokenizer = load_model()

while True:
    output = generate(prompt)
    if output contains final answer:
        break
    if output contains <search>query</search>:
        docs = search(query)
        prompt = prompt + output + format(docs)
```

这就是 Search-R1 的最小 agent loop。

---

## 3. 先看 prompt：系统如何约束模型行为

`infer.py` 里最关键的一段 prompt 如下：

```python
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
```

这段 prompt 规定了四种关键结构：

- `<think>...</think>`：内部推理；
- `<search>...</search>`：搜索动作；
- `<information>...</information>`：搜索返回的外部信息；
- `<answer>...</answer>`：最终答案。

你可以把它理解成一套非常轻量的“交互协议”。

### 3.1 为什么这些标签重要

这些标签不是装饰，而是系统在和模型约定：

- 什么时候你是在“想”；
- 什么时候你是在“发起动作”；
- 什么时候系统会把环境返回给你；
- 什么时候你宣布任务结束。

从 agent 的角度看：

- `<search>` 是 action；
- `<information>` 是 observation；
- `<answer>` 是 terminate signal。

这和我们上一章建立的 RL 视角是完全一致的。

---

## 4. 生成什么时候停下来

如果你不额外做控制，模型可能会一直生成很长一串文本，直到碰到 EOS 为止。

但 Search-R1 的需求不是“让模型一口气写完”，而是：

- 一旦模型完成了一个搜索动作，就先停下来；
- 系统处理完搜索，再继续。

所以 `infer.py` 里定义了一个自定义 stopping criterion：

```python
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
```

直觉上就是：

> 只要模型生成到了 `</search>`，本轮生成就立刻先停住。

### 4.1 这一步为什么必要

因为搜索动作不是语言模型自己能完成的，它需要外部系统介入。

如果不停下来，你就没机会：

- 提取 query；
- 请求检索服务；
- 把结果插回 prompt；
- 再继续下一轮生成。

所以这一步本质上是在把“单次长生成”切成“多轮交互生成”。

---

## 5. 系统如何从输出里提取 query

这个文件里有一个很小但很关键的函数：

```python
def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None
```

它做的事情很简单：

- 在生成文本里找 `<search>...</search>`；
- 如果找到，就取最后一个 query；
- 如果没找到，就返回 `None`。

这里你会看到 Search-R1 的一个基本设计思想：

> 先让模型用文本协议表达动作，再由系统去解析动作。

也就是说，“搜索”虽然看起来写在文本里，但它并不只是普通文本，而是被系统解释为一个可执行动作。

---

## 6. 搜索是怎么真正发生的

### 6.1 请求格式

`infer.py` 里的搜索函数会构造下面的请求：

```python
payload = {
    "queries": [query],
    "topk": 3,
    "return_scores": True
}
results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
```

这说明两件事：

1. Search-R1 的“搜索工具”本质是一个 HTTP 服务；
2. 训练/推理代码和检索系统是解耦的。

这也是 README 中反复强调的设计哲学：

> 主训练流程和搜索引擎服务是分离的。

### 6.2 返回结果会被整理成什么样

代码会把返回文档整理成下面这种字符串：

```python
Doc 1(Title: ... ) ...
Doc 2(Title: ... ) ...
Doc 3(Title: ... ) ...
```

然后再包装进：

```python
<information> ... </information>
```

也就是说，对模型来说，搜索结果不是 Python 对象，也不是 JSON，而是**重新回到文本上下文中的一段 observation**。

这点非常关键，因为这意味着：

- LLM 依然是在做“条件文本生成”；
- 只是上下文中多了一段由外部工具注入的信息。

---

## 7. 整个 while 循环到底在干什么

真正的主循环在这里：

```python
while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(...)

    if outputs[0][-1].item() in curr_eos:
        ...
        break

    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
    if tmp_query:
        search_results = search(tmp_query)
    else:
        search_results = ''

    search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
    prompt += search_text
```

这段代码可以拆成四步：

### 第一步：基于当前 prompt 生成一段新内容

模型此时看到的上下文可能已经包含：

- 原始问题；
- 之前的 `<think>` 内容；
- 之前发起过的 `<search>`；
- 之前返回的 `<information>`。

### 第二步：判断是否已经结束

如果输出已经到 EOS，就认为流程结束。

### 第三步：如果出现 `<search>`，就调用检索服务

这一步把“文本中的动作声明”转成了“真实工具调用”。

### 第四步：把这轮输出和搜索结果拼回 prompt

拼接模板是：

```python
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
```

也就是说，系统会把：

- 模型刚生成的内容；
- 搜到的资料；

一起接回上下文，再进入下一轮。

这就形成了完整闭环。

---

## 8. 用一个具体例子理解这条闭环

假设问题是：

> “某位球员后来成为 KHL 联赛 CSKA Moscow 的总经理，他曾是 Mike Barnett 谈下合同的球员之一，请问是谁？”

模型可能会这样走：

### 第 1 轮

```text
<think>
我需要先确认 Mike Barnett 谈过哪些球员合同，再找其中谁后来当过 CSKA Moscow 总经理。
</think>
<search>Mike Barnett negotiated many contracts player became general manager of CSKA Moscow</search>
```

系统看到 `<search>` 后，去检索服务请求结果。

### 系统注入 observation

```text
<information>
Doc 1(Title: ...) ...
Doc 2(Title: ...) ...
Doc 3(Title: ...) ...
</information>
```

### 第 2 轮

模型再读到这些材料后继续生成：

```text
<think>
从返回结果看，目标人物是 Sergei Fedorov。
</think>
<answer>Sergei Fedorov</answer>
```

这样就结束了。

### 8.1 这里最值得你体会的地方

真正重要的不是答案本身，而是：

- 模型先意识到“自己信息不够”；
- 决定发起搜索；
- 拿到结果后更新判断；
- 最后再回答。

这正是 Search-R1 要训练出来的能力。

---

## 9. 为什么说 `infer.py` 是训练代码的缩影

虽然它不做 RL，但它已经把最本质的交互形式展示出来了。

你后面在 `search_r1/llm_agent/generation.py` 里会看到一个更正式、更 batch 化、更适合训练的版本，但核心思想并没有变：

| `infer.py` 里的概念 | 训练版里的对应概念 |
| --- | --- |
| 当前 prompt | rolling state / input_ids |
| 停到 `</search>` | response postprocess |
| `get_query()` | action parsing |
| `search()` | batch search |
| `<information>` 回填 | next observation |
| while 循环 | `run_llm_loop()` |

所以你完全可以把 `infer.py` 当作后续训练代码的“缩略版草图”。

---

## 10. 这一版实现的局限是什么

理解最小闭环时，也要知道它故意省略了什么。

`infer.py` 没有处理：

- 分布式 rollout；
- 多样本并行；
- reward；
- advantage；
- PPO / GRPO update；
- 非法动作纠正；
- info mask / state masking。

这些更完整的机制都在训练版 `generation.py` 和 `ray_trainer.py` 里。

也正因此，这个文件特别适合“第一次看懂原理”，但不适合“研究完整训练细节”。

---

## 11. 读完这一章后，你应该能回答什么

如果这一章你真的读明白了，你应该已经能回答下面几个问题：

1. Search-R1 为什么不是简单的 RAG？
2. `<search>` 标签为什么不是普通文本，而是一个动作？
3. 为什么系统要在 `</search>` 处截断生成？
4. 搜索结果为什么要重新包装成 `<information>` 放回上下文？
5. 为什么说这套流程天然适合用 RL 训练？

如果这 5 个问题你都能自己复述出来，那么你已经抓住了 Search-R1 最核心的运行机制。

---

## 12. 本章小结

`infer.py` 让你看到：

- Search-R1 的核心不是某个神秘 loss；
- 而是一个**多轮的“生成-搜索-观察-再生成”闭环**；
- 训练代码做的事情，本质上就是把这个闭环扩展成可批量 rollout、可计算 reward、可做 PPO/GRPO 更新的形式。

下一步，如果继续深入，最自然的方向有两个：

1. 去看检索服务到底如何实现；
2. 去看训练主流程如何把这个闭环接进 RL。

这两条都合理；整套讲义选择的是先进入训练主线，等你把 rollout、reward 和 PPO/GRPO 主链路看顺后，再回头把检索系统单独拆开看。这样整体节奏会更顺。