# HFRL (Human Feedback Reinforcement Learning)

SFT, Supervised Fine Tuning과 DPO, Direct Preference Optimization의 과정을 담습니다.

* 주로 QLoRA, Quantized Low-Rank Adaption을 사용하여 어뎁터를 최적화하는 방식으로 진행됩니다.
* SFT > SFT_DPO > SurvLLM 순으로 만들어졌습니다. 먼저 만들어진 코드일수록 하자가 많습니다.
* 현재 모든 구현은 허깅페이스 trl 라이브러리를 이용하여 진행되었습니다. From-Scratch Code는 계획중입니다.