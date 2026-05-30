# 第 04 章：如何把 Search-R1 跑起来，并用 `pdb` 调试

这一章的目标不是追求“完整复现实验结果”，而是帮助你做到两件事：

1. **把代码真的跑起来**；
2. **能自己下断点、单步跟执行流**，从而把前面几章的理解和真实执行过程对上。

对你来说，这一章很重要，因为 Search-R1 不是那种“只读代码就能完全看懂”的项目。它包含：

- 模型生成；
- 检索服务；
- 多轮循环；
- RL 训练调度；
- 分布式 worker。

很多逻辑只有你亲自跑一遍，再配合 `pdb` 看变量，直觉才会真正建立起来。

---

## 1. 先说结论：建议按 3 个层次来跑

我不建议你一上来就直接启动 `train_ppo.sh` 或 `train_grpo.sh`。

更好的顺序是：

### 第 1 层：只跑检索服务
先确认搜索引擎服务本身没问题。

### 第 2 层：只跑 `infer.py`
先看最小闭环：

```text
prompt -> 生成 -> <search> -> 检索 -> <information> -> 再生成
```

### 第 3 层：再进入训练入口
最后再看：

```text
train_ppo.sh -> main_ppo.py -> RayPPOTrainer.fit() -> generation.py
```

这样做的好处是：

- 出问题时更容易定位；
- 调试粒度更可控；
- 你不会被训练基础设施的复杂度一上来淹没。

---

## 2. 你至少要知道的运行组件

从运行角度看，Search-R1 可以拆成 4 块：

### 2.1 Python 训练环境
用于：
- 安装 `verl` / `transformers` / `vllm` / `ray` 等；
- 跑 `infer.py`；
- 跑训练入口 `verl.trainer.main_ppo`。

对应 README 中的推荐环境是：
- Python 3.9
- `torch==2.4.0`
- `vllm==0.6.3`
- `pip install -e .`

可参考：`README.md:67-82`

### 2.2 检索环境
用于：
- 启动本地检索服务 `search_r1/search/retrieval_server.py`；
- 使用 `faiss`、`pyserini`、`fastapi`、`uvicorn`。

可参考：`README.md:84-99`

### 2.3 本地或远程检索器
Search-R1 的搜索工具不是写死在训练进程里的，而是单独暴露一个 HTTP 接口：

- 默认地址：`http://127.0.0.1:8000/retrieve`

这在下面这些位置都能看到：

- `infer.py:67`
- `search_r1/llm_agent/generation.py:452-458`
- `verl/trainer/config/ppo_trainer.yaml:148-150`

### 2.4 数据与模型
如果你只是为了理解执行流，不一定非要先下载完整训练数据。

你可以先分两种目标：

- **目标 A：理解最小闭环** → 优先跑 `infer.py`
- **目标 B：理解训练主流程** → 再准备 parquet 数据和训练脚本

这里要特别区分两类“数据”：

1. **检索数据**：也就是 corpus 和 index，供搜索引擎服务使用；
2. **训练数据**：也就是问答样本，最终会被处理成 `train.parquet` / `test.parquet`。

很多初学者第一次跑这个项目时，会把这两类数据混在一起。其实它们作用完全不同：

- 检索数据决定“模型能搜到什么”；
- 训练数据决定“模型在学什么问题”。

---

## 3. 安装环境：推荐分两个 conda 环境

这是仓库 README 推荐的方式，我也建议你照这个思路来，因为检索侧依赖和训练侧依赖不同。

---

## 4. 环境一：训练 / 推理环境（searchr1）

参考 `README.md:67-82`，你可以用下面的思路搭环境：

```bash
conda create -n searchr1 python=3.9
conda activate searchr1

# 先安装 torch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装 vllm
pip install vllm==0.6.3

# 在仓库根目录安装当前项目
pip install -e .

# 可选：flash attention 2
pip install flash-attn --no-build-isolation

# 便于看日志
pip install wandb
```

### 4.1 为什么 `pip install -e .` 很重要

因为这个仓库是以包形式组织的，`pyproject.toml` 里项目名其实是 `verl`：

- `pyproject.toml:14-24`

而 `setup.py` 会把当前仓库下的包都安装进去：

- `setup.py:37-54`

所以如果你不执行 `pip install -e .`，很多模块导入会出问题，例如：

- `from verl import DataProto`
- `from search_r1.llm_agent.generation import LLMGenerationManager`

### 4.2 如果你只想调试 `infer.py`

至少需要保证这些包能用：

- `transformers`
- `torch`
- `requests`
- 仓库本身（通过 `pip install -e .`）

---

## 5. 环境二：检索环境（retriever）

参考 `README.md:84-99`，可以按下面方式搭建：

```bash
conda create -n retriever python=3.10
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

### 5.1 为什么检索环境建议独立

因为本地检索器依赖：

- `faiss`
- `pyserini`
- 可能还会依赖不同的 CUDA/Java 组合

它和训练/推理环境的依赖重点不同。分开环境后，出问题更容易隔离。

---

## 6. 最小运行路径 A：先只跑检索服务

这是最应该先跑通的一步。

### 6.1 仓库内置启动脚本

你可以直接参考：

- `retrieval_launch.sh:2-13`

脚本本质上等价于：

```bash
python search_r1/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```

### 6.2 启动前你需要准备什么

对于本地 dense retriever，至少需要：

1. 一个 corpus 文件（jsonl）
2. 一个 faiss index 文件
3. 一个 retriever model

README 推荐的下载流程见：

- `README.md:106-123`
- `docs/retriever.md:58-78`

最常见的准备方式是直接使用仓库作者提供的现成资源：

```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

这一步会给你两样关键东西：

- `wiki-18.jsonl`：检索语料库；
- `e5_Flat.index`：对应的 dense index。

如果你只是想理解格式，也可以先看示例语料：

- `example/corpus.jsonl:1-5`

每一行大概长这样：

```json
{"id": "0", "contents": "\"Evan Morris\"\nEvan Morris ..."}
```

这也解释了为什么检索结果里经常会被切成：

- title
- text
- contents

### 6.2.1 如果你想自己准备检索数据

仓库也支持你自己构造语料和索引。

#### 语料格式
README 说明见：
- `README.md:157-201`

你的 corpus 最好是一个 jsonl 文件，每行至少包含：

- `id`
- `contents`

其中 `contents` 推荐写成：

```text
"标题"\n正文
```

#### 自建索引
可以参考：
- `search_r1/search/build_index.sh:2-19`
- `search_r1/search/index_builder.py:297-349`

最小思路是：

```bash
CUDA_VISIBLE_DEVICES=0 python search_r1/search/index_builder.py \
    --retrieval_method e5 \
    --model_path intfloat/e5-base-v2 \
    --corpus_path /your/corpus.jsonl \
    --save_dir /your/index_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat
```

如果你的目标只是先调试 Search-R1，而不是研究 retrieval，本章建议你**优先用作者提供的现成 index + corpus**，因为这样变量更少。

### 6.3 如何验证检索服务是否启动成功

启动成功后，可以用仓库自带的请求脚本测试：

- `search_r1/search/retrieval_request.py:1-23`

也可以自己发一个最小请求：

```python
import requests

payload = {
    "queries": ["What is the capital of France?"],
    "topk": 3,
    "return_scores": True,
}
print(requests.post("http://127.0.0.1:8000/retrieve", json=payload).json())
```

如果这里不通，后面的 `infer.py` 和训练都跑不起来。

---

## 7. 最小运行路径 B：再跑 `infer.py`

这是我最推荐你作为第一个调试入口的文件。

### 7.1 为什么优先调试它

因为它具备这些优点：

- 单文件；
- 无 Ray；
- 无 actor/critic；
- 无 advantage；
- 但已经保留了 Search-R1 的核心交互闭环。

对理解项目来说，它是性价比最高的入口。

### 7.2 直接运行方式

在训练环境中运行：

```bash
conda activate searchr1
python infer.py
```

参考：
- `README.md:142-155`

### 7.3 跑之前你最好先确认两件事

#### 第一，模型路径是否可用

`infer.py` 默认写的是：

- `infer.py:10`

```python
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
```

如果你当前环境没法直接拉这个模型，你可以先改成你本地已有的兼容模型，但要注意：

- prompt 协议仍然要兼容 `<think> / <search> / <information> / <answer>`；
- Qwen 模型默认的 `curr_eos` 也可能需要调整。

#### 第二，本地检索服务必须先启动

因为 `infer.py` 里搜索函数写死调用：

- `infer.py:61-79`

其中请求地址是：

- `infer.py:67`

```python
results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
```

如果服务没起，这里会直接报错。

---

## 8. 如何用 `pdb` 调试 `infer.py`

这部分是最关键的。

### 8.1 最简单方式：直接模块内下断点

你可以直接在 `infer.py` 里临时插入：

```python
import pdb; pdb.set_trace()
```

我建议第一次放在下面几个位置之一。

#### 断点 A：模型生成前
放在 `while True` 里、`model.generate(...)` 之前。

你能看到：
- 当前 `prompt` 长什么样；
- 当前上下文里已经累计了哪些 `<information>`；
- 输入给模型的 token 长度是多少。

#### 断点 B：拿到 `output_text` 之后
对应位置大致是：
- `infer.py:115-118`

你能看到：
- 这一轮模型到底生成了什么；
- 有没有出现 `<search>`；
- 生成结果是偏 reasoning 还是偏 answer。

#### 断点 C：`tmp_query = get_query(...)` 之后
对应位置大致是：
- `infer.py:118-123`

你能看到：
- query 是否成功提取；
- 提取结果是否合理；
- query 为什么有时为空。

#### 断点 D：`search_results = search(tmp_query)` 之后
你能看到：
- 检索服务返回的字符串到底长什么样；
- title / text 是怎样被拼进 observation 的；
- 搜到的材料是否真的有帮助。

#### 断点 E：`prompt += search_text` 之后
你能看到：
- 新一轮 prompt 变成了什么；
- 为什么下一轮模型会基于这些 observation 继续生成。

### 8.2 推荐的 `pdb` 观察命令

在 `pdb` 里，你最常用到这些命令就够了：

```text
n   # 下一行
s   # 进入函数
c   # 继续运行到下一个断点
p x # 打印变量 x
pp x # 更好看的打印
l   # 看附近代码
w   # 看调用栈
q   # 退出
```

第一次调试时，我建议你重点打印：

```python
p prompt[-1000:]
p output_text
p tmp_query
p search_results[:500]
```

这样你能很快把“代码行为”和“文本协议”对应起来。

---

## 9. 更进一步：调试检索服务 `retrieval_server.py`

如果你已经跑通 `infer.py`，下一步我建议调试检索服务本身。

### 9.1 为什么值得调这个文件

因为 Search-R1 的搜索能力有两层：

1. 模型会不会发起搜索；
2. 搜索服务返回的结果质量如何。

如果第 2 层有问题，你会误以为是模型不行，但其实可能是 retriever 没配好。

### 9.2 推荐断点位置

文件：
- `search_r1/search/retrieval_server.py`

建议断在：

#### 断点 A：`retrieve_endpoint()` 入口
对应：
- `retrieval_server.py:326-358`

你能看到：
- 请求体 `queries` 到底是什么；
- `topk` 是否符合预期；
- 是否真的传入了多个 query。

#### 断点 B：`retriever.batch_search(...)` 调用前后
对应：
- `retrieval_server.py:341-345`

你能看到：
- 检索前后的结果结构；
- scores 是否存在；
- 返回的 doc 是否带 `contents`。

### 9.3 DenseRetriever 的推荐断点

文件：
- `search_r1/search/retrieval.py`

建议断在：

#### 断点 C：`DenseRetriever._batch_search()`
对应：
- `retrieval.py:274-309`

看这些变量：

```python
p query_list[:3]
p num
p batch_idxs[:1]
p batch_scores[:1]
```

你会看清：
- query 是如何被编码成 embedding 的；
- faiss 返回的是哪些 doc idx；
- top-k 是如何变成最终文档列表的。

---

## 10. 最小运行路径 C：进入训练，但先不要追求完整大训练

如果你的目标是“理解训练主流程”，我建议你先把训练理解成**可调试入口**，而不是立刻完整跑通大规模实验。

### 10.1 训练入口在哪里

真正入口是：

- `train_ppo.sh:29-90`
- `train_grpo.sh:29-82`

这两个脚本最终都会调用：

```bash
python3 -m verl.trainer.main_ppo
```

也就是：
- `verl/trainer/main_ppo.py`

### 10.2 PPO 与 GRPO 的入口差异

看脚本就能很直观看到：

- PPO：`train_ppo.sh:41` 设置 `algorithm.adv_estimator=gae`
- GRPO：`train_grpo.sh:41` 设置 `algorithm.adv_estimator=grpo`

其他差异还包括：

- PPO 在当前训练主路径里会用 critic：`train_ppo.sh:61-69`
- GRPO 的 advantage 计算主路径不依赖 critic，并会设置 actor 侧 KL loss：`train_grpo.sh:47-64`
- GRPO 让 `n_agent=5`：`train_grpo.sh:62`

另外还有一个很实操、但容易踩坑的点：

- `train_grpo.sh` 文件前面只定义了 `DATA_DIR`，但真正命令里写的是 `$TRAIN_DATA_DIR` 和 `$TEST_DATA_DIR`（见 `train_grpo.sh:30-31`）。

也就是说，如果你直接照当前脚本原样运行 GRPO，很可能需要先手动修正这两个变量，或者把它们显式导出好。

> 训练入口层面，PPO 和 GRPO 的主要差别体现在 shell override 里；但当前 GRPO 脚本本身还带着一个数据路径变量不一致的小坑。

---

## 11. 训练前的数据准备要知道什么

如果你真要启动训练，至少需要 parquet 数据。

### 11.1 训练数据到底从哪里来

这一点确实应该明确写出来。

这个项目里，训练数据有两种常见来源：

#### 来源 A：直接下载作者整理好的数据集
这个方式最省事，适合你想尽快进入调试训练主流程的时候。

参考：
- `scripts/nq_hotpotqa/README.md:4-8`

命令是：

```bash
huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train --local-dir $WORK_DIR/data/nq_hotpotqa_train
```

这适合“想尽快跑通官方流程”的场景。

#### 来源 B：自己从原始问答数据处理生成 parquet
这个方式更适合理解数据是怎么进入训练的。

参考：
- `README.md:114-117`
- `scripts/data_process/nq_search.py:42-101`

命令是：

```bash
python scripts/data_process/nq_search.py
```

这个脚本会从：

- `RUC-NLPIR/FlashRAG_datasets` 的 `nq` 子集

读取原始样本，然后生成：

- `data/nq_search/train.parquet`
- `data/nq_search/test.parquet`

所以从“理解代码执行流”的角度看，我其实更推荐你至少跑一次 `nq_search.py`，因为这样你会真正知道训练样本是怎么组织出来的。

### 11.2 仓库怎么把原始问答样本变成训练样本

脚本在：
- `scripts/data_process/nq_search.py:42-101`

它会把原始样本变成类似下面的结构：

```python
data = {
    "data_source": data_source,
    "prompt": [{"role": "user", "content": question}],
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

然后写成：

- `train.parquet`
- `test.parquet`

### 11.3 这份训练数据里最值得你关心的字段

如果你的目标是调试训练过程，我建议最先盯住下面四个字段：

- `prompt`
- `reward_model.ground_truth`
- `data_source`
- `extra_info.index`

它们分别决定：

- 模型看到的问题是什么；
- 最终 reward 如何对答案打分；
- 当前样本来自哪个任务；
- 这一题在 GRPO 中如何分组。

### 11.4 为什么这里的 `index` 很重要

因为 GRPO 会用同题多样本分组比较。

这个字段会一路进入：

- `verl/utils/dataset/rl_dataset.py:151-154`
- 再到 `ray_trainer.py:744-748`
- 再到 `compute_grpo_outcome_advantage(...)`

如果后面你调 GRPO，这个字段非常值得盯着看。

### 11.5 如果你只是为了调试，需要把数据准备到什么程度

这也是很实际的问题。

#### 情况 A：你只想调 `infer.py`
那你**不需要训练数据 parquet**。

你只需要：
- 一个可用模型；
- 一个可用检索服务；
- 对应的 corpus/index。

#### 情况 B：你想调 `main_ppo.py` 或 `ray_trainer.py`
那你至少需要：
- `train.parquet`
- `test.parquet`
- 检索服务能正常返回结果

#### 情况 C：你想尽快进入训练调试，而不想在数据处理上花太多时间
那就优先：
1. 下载作者整理好的数据集；
2. 配好 `train_ppo.sh` / `train_grpo.sh` 里的数据路径；
3. 再开始打断点。

也就是说：

> 最小闭环调试主要依赖“模型 + 检索数据”；训练主流程调试才依赖“parquet 训练数据”。

---

## 12. 如何用 `pdb` 调试训练入口

这部分我建议你分三层断点。

### 12.1 第一层：断在 `main_ppo.py`

文件：
- `verl/trainer/main_ppo.py`

建议断点：

#### 断点 A：`main_task(config)` 一开始
对应：
- `main_ppo.py:113-122`

这里重点看：

```python
p config.actor_rollout_ref.model.path
p config.data.train_files
p config.retriever.url
p config.algorithm.adv_estimator
```

作用：
- 确认 shell override 是否真的生效；
- 确认你以为传进去的路径，程序实际看到的是不是那个路径。

#### 断点 B：`reward_fn = RewardManager(...)` 之后
对应：
- `main_ppo.py:183-196`

作用：
- 明确 trainer 真实拿到的 reward function 是什么；
- 为后面调 reward 做入口。

### 12.2 第二层：断在 `RayPPOTrainer.fit()`

文件：
- `verl/trainer/ppo/ray_trainer.py`

这是最值得调试的地方。

建议断点：

#### 断点 C：batch 刚从 dataloader 读出来时
对应：
- `ray_trainer.py:695-705`

看这些变量：

```python
p batch_dict.keys()
p batch.non_tensor_batch['data_source'][:3]
p batch.non_tensor_batch['index'][:3]
p batch.batch['input_ids'].shape
```

你能搞清：
- 一个训练 batch 到底长什么样；
- prompt 和非 tensor 信息是怎么共存的。

#### 断点 D：`generation_manager.run_llm_loop(...)` 前后
对应：
- `ray_trainer.py:725-748`

这一步最关键，因为它就是 Search-R1 相比普通 PPO 的核心增量。

重点看：

```python
p first_input_ids.shape
p final_gen_batch_output.batch.keys()
p final_gen_batch_output.meta_info.keys()
```

#### 断点 E：reward 计算处
对应：
- `ray_trainer.py:778-805`

重点看：

```python
p batch.batch['token_level_scores'].shape
p batch.batch['token_level_rewards'].shape
p batch.batch['advantages'].shape
```

这一步能把 rollout、reward、advantage 三者串起来。

### 12.3 第三层：断在 `generation.py`

文件：
- `search_r1/llm_agent/generation.py`

这是训练时最像“环境 step”的地方。

建议断点：

#### 断点 F：`run_llm_loop()` 开始
对应：
- `generation.py:220-233`

看：

```python
p gen_batch.batch.keys()
p initial_input_ids.shape
p self.config.max_turns
```

#### 断点 G：`execute_predictions(...)` 前后
对应：
- `generation.py:252-265`

看：

```python
p responses_str[:2]
p next_obs[:2]
p dones[:5]
p valid_action[:5]
p is_search[:5]
```

这一步会让你非常直观地看到：

- 哪些输出被识别为 search；
- 哪些输出被识别为 answer；
- 非法输出会如何被修正。

#### 断点 H：`batch_search()` 内部
对应：
- `generation.py:438-458`

看：

```python
p queries
p payload
```

这样你能看到训练时发给检索服务的真实 query 长什么样。

---

## 13. 训练调试时，先选 PPO 还是 GRPO？

如果你的目标是**更容易调通执行流**，我建议：

### 第一阶段先调 PPO
原因：
- 路径更“标准”；
- 一题一条 rollout，更容易理解；
- 你更容易把 reward、value、advantage 的关系对起来。

### 第二阶段再调 GRPO
原因：
- 它会引入同题多样本；
- `n_agent=5` 会让数据流更复杂；
- 但更适合理解“组内相对比较”的思想。

也就是说：

> 想先建立主线理解，先调 PPO；想理解 Search-R1 对 GRPO 的利用方式，再进 GRPO。

---

## 14. 你最值得观察的 12 个变量

如果你调试时不知道该盯什么，我建议优先看这 12 个变量：

### 推理 / 搜索阶段
1. `prompt`
2. `output_text`
3. `tmp_query`
4. `search_results`
5. `search_text`

### 训练阶段
6. `batch.batch['input_ids']`
7. `batch.non_tensor_batch['index']`
8. `final_gen_batch_output.batch['responses']`
9. `batch.batch['token_level_scores']`
10. `batch.batch['token_level_rewards']`
11. `batch.batch['advantages']`
12. `batch.meta_info`

只要这 12 个变量你能跟住，Search-R1 的主流程你就已经掌握大半了。

---

## 15. 一个很实用的调试策略：先只看“文本”，再看“张量”

你是算法背景，看到张量通常不会害怕，但这里我还是建议一个顺序：

### 第一轮：只看文本层
例如：
- prompt 长什么样；
- query 长什么样；
- observation 长什么样；
- answer 长什么样。

### 第二轮：再看张量层
例如：
- `input_ids.shape`
- `responses.shape`
- `attention_mask.shape`
- `advantages.shape`

为什么这样更好？

因为 Search-R1 首先是一个**文本协议驱动的 agent 系统**，其次才是一个张量计算图。如果一上来全盯 tensor shape，很容易丢掉语义层面的直觉。

---

## 16. 如果你现在就要开始，我建议你的实际操作顺序

你可以直接照这个顺序来：

### 第一步：起检索服务
- 先准备 index / corpus
- 运行 `retrieval_launch.sh`
- 用 `retrieval_request.py` 验证 `/retrieve` 正常

### 第二步：调试 `infer.py`
- 先正常跑一次
- 再在 `get_query()`、`search()`、`prompt += search_text` 附近下断点
- 观察 1~2 轮搜索闭环

### 第三步：调试 `main_ppo.py`
- 重点看 config 是否正确
- 不要一开始就深挖 actor/critic 内部实现

### 第四步：调试 `RayPPOTrainer.fit()`
- 重点看 batch、rollout、reward、advantage

### 第五步：进入 `generation.py`
- 重点看多轮 agent loop
- 真正理解 Search-R1 的核心增量

这个顺序非常适合“以理解为目标”的阅读，而不是“以复现实验数值为目标”。

---

## 17. 本章小结

这一章最重要的不是具体命令本身，而是下面这套方法：

1. **先把检索服务跑通**；
2. **先用 `infer.py` 看最小闭环**；
3. **再进入训练入口调主流程**；
4. **最后进入 `generation.py` 看 Search-R1 的核心 agent 逻辑**。

对你来说，最好的第一调试入口不是大训练脚本，而是：

> `infer.py` + `retrieval_server.py`

等你把这两个入口跟顺了，再去看 `main_ppo.py` 和 `ray_trainer.py`，会轻松很多。
