# SFT + DPO Overview

데이터 구축부터 SFT + DPO 과정을 전부 포함합니다. 데이터 구축 과정은 데이터별로 별도의 처리가 필요하며, 여기서 제시된 코드는 예시입니다. 깃허브에 지속적으로 업데이트될 예정입니다.

https://github.com/HollyRiver/HFRL/tree/main/SFT_DPO

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