### 3.2 数据格式
```
字段要求：

- `messages`
  标准对话格式，最后一条必须是正例 assistant 回复
- `rejected_response`
  `List[str]`，表示同一个 prompt 下的多个负例

也兼容单负例字符串：

```json
{"messages":[...],"rejected_response":"单个负例"}
```

但这次改动的重点是一正多负，也就是：

```json
"rejected_response": ["neg1", "neg2", "neg3"]
```




cd /root/autodl-tmp/ms-swift-edit/ms-swift

env -u OMP_NUM_THREADS \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/autodl-tmp/ms-swift-edit/ms-swift \
python swift/cli/rlhf.py \
  --rlhf_type dpo \
  --model /root/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct \
  --train_type lora \
  --dataset tmp_multi_neg_dpo_local \
  --custom_dataset_info /root/autodl-tmp/ms-swift-edit/ms-swift/tmp_multi_neg_dpo_dataset_local.json \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --lora_rank 4 \
  --lora_alpha 8 \
  --target_modules all-linear \
  --max_length 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --logging_steps 1 \
  --save_steps 1 \
  --save_total_limit 1 \
  --output_dir /root/autodl-tmp/output/dpo_smoke
```



```bash
/root/autodl-tmp/output/dpo_smoke/.../checkpoint-1
```

这说明：

- 模板编码支持一正多负
- 多负例 DPO loss 已经进入训练链路

### 3.6 正式训练时怎么改

如果你不是只做 smoke test，而是正式训练，通常至少要改这些参数：

- `--max_steps`
- `--num_train_epochs`
- `--max_length`
- `--per_device_train_batch_size`
- `--learning_rate`
- `--output_dir`

如果你想继续用全参数训练，也可以把：

```bash
--train_type lora
```

改成：

```bash
--train_type full
```

但显存占用会明显上升。

## 4. GRPO Rule Reward 教程

### 4.1 这个改动支持什么

当前 GRPO 改动支持：

- 数据集中每条样本自带规则代码
- 运行时逐条执行规则代码
- 根据规则打 reward
- 支持多规则聚合

关键改动文件：

- `swift/llm/argument/rlhf_args.py`
- `examples/train/grpo/plugin/rule_code_reward_plugin.py`
- `examples/train/grpo/plugin/run_rule_code_reward.sh`

### 4.2 新增参数

当前新增了一个 CLI 参数：

```python
rule_reward_aggregation: Literal['mean', 'all_or_nothing'] = 'mean'
```

可用值：

- `mean`
  多条规则的 reward 取平均
- `all_or_nothing`
  只要有一条规则失败，最终 reward 就是 `0.0`

### 4.3 数据格式

你的数据文件是：

```bash
/root/autodl-tmp/ms-swift-edit/data_grpo/train_rule_rl1_1204_overlap_with_dpo_0114.json
```

虽然扩展名是 `.json`，但实际内容是 `jsonl`，也就是每行一个 JSON。

单条样本结构大致如下：

```json
{
  "query": "Provide 5 tongue twister sentences about famous ballet dancers.",
  "code": {
    "code": [
      "def check_following(response):\n    return 'ballet' in response.lower()"
    ],
    "constrain": [
      "回复必须满足规则"
    ]
  },
  "augmented_rewrite": "..."
}
```

字段说明：

- `query`
  用户输入
- `code.code`
  `List[str]`，每个元素都是一段 Python 代码，并且必须定义：

```python
def check_following(response): ...
```

- `code.constrain`
  规则的自然语言说明，不参与打分
- `augmented_rewrite`
  额外改写字段，reward 不依赖它

### 4.4 Reward 是怎么计算的

插件位置：

```bash
examples/train/grpo/plugin/rule_code_reward_plugin.py
```

处理流程：

1. 读取样本里的 `code.code`
2. 每段代码单独 `exec`
3. 提取 `check_following`
4. 用模型生成的 completion 调用这个函数
5. 每条规则：
   - 通过记 `1.0`
   - 失败记 `0.0`
   - 抛异常也记 `0.0`
6. 再根据 `rule_reward_aggregation` 聚合成最终 reward

### 4.5 最小可运行命令

下面这条命令已经在当前机器上实际跑通：

```bash
cd /root/autodl-tmp/ms-swift-edit/ms-swift

env -u OMP_NUM_THREADS \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/autodl-tmp/ms-swift-edit/ms-swift \
python swift/cli/rlhf.py \
  --rlhf_type grpo \
  --model /root/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B-Instruct \
  --train_type lora \
  --dataset /root/autodl-tmp/ms-swift-edit/data_grpo/train_rule_rl1_1204_overlap_with_dpo_0114.json \
  --external_plugins /root/autodl-tmp/ms-swift-edit/ms-swift/examples/train/grpo/plugin/rule_code_reward_plugin.py \
  --reward_funcs dataset_rule_code_reward \
  --rule_reward_aggregation mean \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --lora_rank 4 \
  --lora_alpha 8 \
  --target_modules all-linear \
  --max_length 512 \
  --max_completion_length 64 \
  --num_generations 2 \
  --generation_batch_size 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --logging_steps 1 \
  --save_steps 1 \
  --save_total_limit 1 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --output_dir /root/autodl-tmp/output/grpo_smoke
```

### 4.6 GRPO 这里最容易踩的坑

#### 1. `generation_batch_size` 必须能整除 `num_generations`

这次实际测试里，如果设置：

```bash
--num_generations 2
```

但没有设置合适的：

```bash
--generation_batch_size
```

会直接报错：

```text
generation_batch_size (1) must be divisible by num_generations (2)
```

所以最小可跑配置至少要保证：

```bash
--num_generations 2 \
--generation_batch_size 2
```

#### 2. 规则代码必须定义 `check_following`

也就是每一段规则代码都必须至少包含：

```python
def check_following(response): ...
```

否则插件会报错并把该规则记为 `0.0`。

#### 3. 规则代码执行异常会被记 0 分

如果规则代码内部抛异常，当前实现不会中断整个训练，而是：

- 打 warning
- 当前规则记为 `0.0`

### 4.7 跑通时你会看到什么

训练正常时会进入：

```text
Train: 100%
```

并输出类似指标：

```text
'rewards/DatasetRuleCodeReward/mean': 1.0
```

然后保存 checkpoint：

```bash
/root/autodl-tmp/output/grpo_smoke/.../checkpoint-1
```

这说明：

- 外部 reward 插件已加载成功
- `rule_reward_aggregation` 参数已进入 reward 类
- 规则代码 reward 已参与训练

## 5. 两种训练方式的区别

### Multi-Negative DPO

适合：

- 你已经有明确的正例回答
- 同时也有同一 prompt 下的多个负例回答
- 你希望模型学会“正例优于多种负例”

特点：

- 监督信号来自 pair / list-wise 偏好数据
- 不需要在线生成 reward

### GRPO Rule Reward

适合：

- 你没有现成的偏好对
- 但你能写出可执行规则
- 希望模型通过 rollout + reward 的方式优化输出

特点：

- reward 来自规则函数
- 训练时需要生成 completion 再打分

## 6. 建议的使用顺序

如果你现在只是想验证改动没问题，建议按这个顺序：

1. 先跑本文档里的 DPO 1-step 命令
2. 再跑本文档里的 GRPO 1-step 命令
3. 确认都能生成 checkpoint
4. 再逐步放大：
   - 模型
   - batch size
   - max_length
   - max_steps
   - 正式数据量

## 7. 常用检查命令

### 看 GPU

```bash
nvidia-smi
```

### 做语法检查

```bash
cd /root/autodl-tmp/ms-swift-edit/ms-swift

python -m py_compile \
  swift/llm/template/base.py \
  swift/trainers/rlhf_trainer/dpo_trainer.py \
  swift/megatron/trainers/dpo_trainer.py \
  swift/llm/argument/rlhf_args.py \
  examples/train/grpo/plugin/rule_code_reward_plugin.py
```

### 查看输出目录

```bash
find /root/autodl-tmp/output/dpo_smoke -maxdepth 3 -type d | sort
find /root/autodl-tmp/output/grpo_smoke -maxdepth 3 -type d | sort
```

## 8. 当前机器上与教程相关的关键文件

- `readme.md`
- `tmp_multi_neg_dpo.jsonl`
- `tmp_multi_neg_dpo_dataset.json`
- `tmp_multi_neg_dpo_dataset_local.json`
- `examples/train/grpo/plugin/rule_code_reward_plugin.py`
- `examples/train/grpo/plugin/run_rule_code_reward.sh`
- `swift/llm/template/base.py`
- `swift/trainers/rlhf_trainer/dpo_trainer.py`
- `swift/megatron/trainers/dpo_trainer.py`
- `swift/llm/argument/rlhf_args.py`


