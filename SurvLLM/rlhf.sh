## 여기 있는거 한번에 다 안돼요. 성능상 한번에 해도 안되고.
## 그리고 실행할 때에는 &&가 아닌 &를 붙여야 합니다. 여긴 그냥 순차적으로 된다는 가정하에 작성했어요.

nohup python csv_to_json_dataset.py --target="data/data_sample1_for_SFT_20260205.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt" &&

nohup python SFT.py --config config/SFT_config_v1.1.2.yaml > logs/sft_log_v1.1.2.txt &&
## multi GPU 사용 시 fsdp_config_qlora.yaml 파일에서 num_processes에 GPU 숫자만 수정하여 아래를 전부 입력
# nohup env \
# accelerate launch --config_file "config/fsdp_config_qlora.yaml" \
# SFT.py --config config/SFT_config_multi_GPU.yaml > sft_test.log &

## generated_data for DPO를 생성
nohup python csv_to_json_dataset.py --target="data/gen_data_for_dpo_20260205.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt" &

## 양자화 모델 생성
nohup python gen_llama_nf4.py &&

## SFT에서 온전한 모델을 픽스하고, 양자화된 base model이 따로 저장되었으며, 추론에 사용할 프롬프트가 준비되었을 때
## !!!어댑터 저장 폴더에는 무조건 "sft"라는 키워드를 넣어주세요, 그래야 인식합니다!!!
nohup python vllm_inference.py --base_model_path="base_model/Llama-3.1-8B-Instruct-nf4"\
                               --adapter_path="adapter/Zip-Llama-sft-v1.1.2"\
                               --inference_data="data/gen_data_for_dpo_20260205.json"\
                               --output_dir="data/generated_data_v1.1.2.csv"\
                               --gen_nums=5\
                               --sampling=True\
                               --temperature=1.0\
                               --repetition_penalty=1.0\
                               --gpu_memory_util=0.9\
                               --seed=42 &

## SFT에서 온전한 모델을 픽스하고, 데이터셋이 준비되었을 때
nohup python csv_to_json_dataset.py --target="data/data_sample_max_for_DPO_20260205.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt" &&

nohup python DPO.py --config config/DPO_config_v1.1.0.yaml > logs/dpo_log_v1.1.0.txt &&
## Adapter Twice Load로 일단 되긴 하는데, 메모리 더 많이 먹음...
# nohup env \
# NCCL_TIMEOUT=600 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --config_file "config/fsdp_config_qlora_dpo.yaml" \
# DPO.py --config config/DPO_config_multi_GPU.yaml > dpo_test.log &

nohup python csv_to_json_dataset.py --target="data/data_all.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt" &&

## DPO에서 온전한 모델을 픽스하고, 양자화된 base model이 따로 저장되었으며, 추론에 사용할 프롬프트가 준비되었을 때
nohup python vllm_inference.py --base_model_path="base_model/Llama-3.1-8B-Instruct-nf4"\
                               --adapter_path="adapter/Zip-Llama-aligned-v1.1.1"\
                               --inference_data="data/data_all.json"\
                               --output_dir="data/inference_all.csv"\
                               --sampling=True\
                               --repetition_penalty=1.0\
                               --gpu_memory_util=0.9\
                               --seed=42 &