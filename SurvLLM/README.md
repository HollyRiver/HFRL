# 생존 분석을 위한 텍스트 언어 모델 튜닝

* [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 모델을 튜닝하여 장문의 텍스트에서 중요한 단서를 1차적으로 추출
* [QLoRA](https://arxiv.org/abs/2305.14314), [Load to adapter twice](https://huggingface.co/docs/trl/dpo_trainer#using-option-3---load-the-adapter-twice) 사용


## Setup
* 100GB의 VRAM 및 200GB의 CPU RAM (권장). 입력되는 최대 시퀀스 길이를 더 길게 설정한다면 이보다 많이 필요합니다. (해당 문서에서는 16,384)
* 구축된 아나콘다 환경 (cuda 12.8, Ubuntu 20.04에서 구동시켰으나, Ubuntu 22.04 이상을 권장합니다. Ubuntu 20.04 버전에서는 flash-attention 실행을 위한 다운그레이드 및 GLibc 업데이트가 필요합니다.)
* Dependencies installation: `pip install transformers bitsandbytes datasets sentencepiece accelerate trl peft wandb openai pqdm`, pytorch와 flash-attention, vllm 설치는 부가적으로 수행해주세요. flash-attention-2가 사용되었습니다.

```
conda env create -f LLM.ymal
conda activate LLM
```

## SFT 실험

### 1차 실험 결과 (1e-4)

* test dataset의 예측된 값들은 마지막 수치를 나이브하게 가져오고 있다는 것을 확인. 과적합될수록 해당 경향은 강해짐
* eval loss는 계속해서 증가하였음
* Learning rate를 조정할 필요가 있는듯 -> 초반부 결과는 조금 나았음. 과적합됨

### 2차 실험 결과 (1e-5)

* 초반에 손실이 많이 줄어들고 이후 수렴하는 형태
* 15 epoch / 35 epoch / 40 epoch의 성능이 나쁘지 않았음
* 전반적으로 형태는 유사하게 따라가려 하고 있음. 다만, 수치를 끌어올 때 대푯값이 아닌 가장 마지막에 기재된 수치를 가져오는 경향이 있음. 과적합의 영향 또는 온도 설정의 여파로 사료됨
> 온도의 문제인지 확인하기 위해 빔 서치로 추론한 결과와 비교해볼 필요 있음

### 3차 실험 (5e-6)

* 수치값 기준 50/40/30 epoch가 거의 동일, 가장 성능 좋았음. 50 epoch이 가장 훌륭함 (정성적 평가 안함)
* 50 epoch이 지금껏 중에 가장 훌륭함 (수치값 기준. 정성적 평가 안함)

### 4차 실험 (2e-6)

* 3차 실험과 거의 동일한 결과. 3차 실험에서의 모델을 사용하는 것이 나을듯


### 기타

* 시스템 프롬프트에 어떤 값을 가져와야 하는지를 조금 더 명시해야 한다고 판단됨
* 훈련 데이터셋 규모가 너무 작음. 일반화에 어려움을 겪을 가능성 높음
* text 데이터에 체온 화씨/섭씨 혼용되고 있음 -> 섭씨 온도 출력: 34.0°C
* 적어도 수치값에 한정해서는 없는 값을 지어내서 제시하지는 않는듯
* Glucose를 많이 틀림

## DPO 실험

### 기타 (https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997)

* 병합 이후 다시 양자화하여 DPO adpater를 부착, 학습할 경우 SFT 모델이 왜곡됨. 따라서 모델을 병합하지 않고, SFT adapter를 DPO로 튜닝하는 것이 가장 효과적인 방법임.
* 최종 DPO를 마친 모델은 병합하여 추론하는 것이 바람직하나, 여러 시도를 해봤음에도 결과가 왜곡되었음. 스크래치로 구현하면 방법이 있을지도... 참고 문서가 거의 없음.
* QLoRA로 학습 후 어뎁터와 병합할 경우, 결과가 뭉개짐. 아직 효과적인 QLoRA merge를 지원하는 공식적인 방법은 없?음.
* 추론 및 빌드는 어뎁터를 병합하지 않았더라도 vllm을 무조건 활용하세요. 처음 컴파일 하는데 시간이 많이 걸리긴 하지만, cpp 기반 + 유동 배치 활용이라는 점에서 몇백배는 더 빠른 퍼포먼스를 보여줍니다.