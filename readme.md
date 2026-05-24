# STAIF: Stage-wise Optimization for Complex Instruction Following

This repository contains the code and dataset for the paper: **"Following complex instructions with multiple explicit constraints remains a fundamental challenge for large language models."**


## How to Run

### 0. Preparation

First, navigate to the `ms-swift` directory:

cd ./ms-swift

#### Data Format for Stage 1
For the multi-negative DPO in Stage 1, your JSON dataset must contain a list of multiple negative responses under the `rejected_response` key. Example format:

```json
{
  "query": "Your complex instruction here...",
  "chosen_response": "The good response...",
  "rejected_response": ["neg1", "neg2", "neg3"]
}
```

### 1. Stage 1: Soft Constraint Alignment (DPO)

In this stage, we use DPO with multiple negative samples to align the model's sensitivity to subjective/soft constraints.

Run the following script:

python swift/cli/rlhf.py \
  --rlhf_type dpo \
  --model  \
  --train_type full \
  --dataset   \
  --custom_dataset_info  \
  --torch_dtype bfloat16 \
  --max_length 4096\
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --max_steps  \
  --logging_steps 1 \
  --save_steps  \
  --save_total_limit  \
  --output_dir 

### 2. Stage 2: Hard Constraint Enforcement (RLVR / GRPO)

In Stage 2, we fine-tune the model from Stage 1 using Group Relative Policy Optimization (GRPO) with rule-based/verifiable rewards to enforce strict compliance with hard constraints.

Run the following script:

python swift/cli/rlhf.py \
  --rlhf_type grpo \
  --model  \
  --train_type full \
  --dataset  \
  --external_plugins ./ms-swift/examples/train/grpo/plugin/rule_code_reward_plugin.py \
  --reward_funcs dataset_rule_code_reward \
  --rule_reward_aggregation mean \
  --torch_dtype bfloat16 \
  --max_length 4096 \
  --num_generations 16 \
  --generation_batch_size 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs  
  
### STAINSTRUCT Dataset
To support this method, we introduce **STAINSTRUCT**, a high-quality bilingual (English, Chinese) dataset.**Note on Dataset Availability:** Due to upload size limitations, in this release we have open-sourced the majority of the preprocessed data for Stage 1 and a portion of the data for Stage 2. We will open-source the complete training data for Stage 2 in the near future. All currently released data can be found in the `./data/` directory.
