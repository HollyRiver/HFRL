# SFT + DPO Overview

## Setup
* 20GB의 VRAM 및 48GB의 CPU RAM (권장)
* 구축된 아나콘다 환경 (cuda 12.8, Ubuntu 22.04 이상 추천)

```
conda env create -f LLM.ymal
conda activate LLM
```

## Full Training Examples

```
# data setting example
nohup python dataset_setting.py &

# sft training example
nohup python SFT.py --config config/SFT_config.yaml &

# dpo traning example
nohup python DPO.py --config config/DPO_config.yaml &
```