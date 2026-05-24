# STAIF: Stage-wise Optimization for Complex Instruction Following

This repository contains the code and dataset for the paper: **"Following complex instructions with multiple explicit constraints remains a fundamental challenge for large language models."**

###  STAINSTRUCT Dataset
To support this method, we introduce **STAINSTRUCT**, a high-quality bilingual (English, Chinese) dataset containing approximately **31,000** complex multi-constraint instructions. Extensive analyses validate the design of STAIF, showing state-of-the-art performance on representative benchmarks against strong baselines, as well as genuine generalization.

---

## How to Run

### 0. Preparation

First, navigate to the `ms-swift` directory:

```bash
cd ./ms-swift
```

#### Data Format for Stage 1
For the multi-negative DPO in Stage 1, your JSON dataset must contain a list of multiple negative responses under the `rejected_response` key. Example format:

```json
{
  "query": "Your complex instruction here...",
  "chosen_response": "The good response...",
  "rejected_response": ["neg1", "neg2", "neg3"]
}
```

---

### 1. Stage 1: Soft Constraint Alignment (DPO)

In this stage, we use DPO with multiple negative samples to align the model's sensitivity to subjective/soft constraints.

Run the following script:

```bash
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

---
### 2. Stage 2: Hard Constraint Enforcement (RLVR / GRPO)

In Stage 2, we fine-tune the model from Stage 1 using Group Relative Policy Optimization (GRPO) with rule-based/verifiable rewards to enforce strict compliance with hard constraints.


python swift/cli/rlhf.py \
  --rlhf_type grpo \
  --model <path_to_stage1_model> \
  --train_type lora \
  --dataset <your_grpo_dataset> \
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
