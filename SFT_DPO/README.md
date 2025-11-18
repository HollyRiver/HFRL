# SFT + DPO Overview

데이터 구축부터 SFT + DPO 과정을 전부 포함합니다. 데이터 구축 과정은 데이터별로 별도의 처리가 필요하며, 여기서 제시된 코드는 예시입니다. 깃허브에 지속적으로 업데이트될 예정입니다.

## Setup
* 20GB의 VRAM 및 48GB의 CPU RAM (권장)
* 구축된 아나콘다 환경 (cuda 12.8, Ubuntu 20.04에서 구동시켰으나, Ubuntu 22.04 이상을 권장합니다. Ubuntu 20.04 버전에서는 flash-attention 실행을 위한 다운그레이드 및 GLibc 업데이트가 필요합니다.)
* Dependencies installation: `pip install transformers bitsandbytes accelerate datasets sentencepiece trl peft wandb openai pqdm`, pytorch와 flash-attention은 먼저 따로 설치해주세요. 종속성을 고려해야 합니다.

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

## 기타

* legacy가 붙은 파일은 잘못된 파일입니다. 일부 코드만 수정하면 일반 LoRA에서 활용될 여지가 있어 남겨두나, 수정 없이 절대로 사용하지 말아주세요. 어뎁터 로드 과정과 훈련 모듈 설정 자체가 잘못되었습니다.