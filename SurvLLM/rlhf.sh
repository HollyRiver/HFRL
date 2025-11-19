nohup python sft_dataset_setting.py &&

nohup python SFT.py --config config/SFT_config.yaml > sft_log.txt &&

## SFT에서 온전한 모델을 픽스하고, 해당 어뎁터를 삽입
## temperature 설정은 1.0 정도로 해야 다양한 결과 나옴
nohup python sft_generate.py --adapter_name="adapter/Zip-Llama-sft" --output_name="gen_data.csv" --gen_nums=5 --temp=1.0 &&

## SFT에서 온전한 모델을 픽스하고, 데이터셋이 준비되었을 때
nohup python dpo_dataset_setting.py &&

nohup python DPO.py --config config/DPO_config.yaml > dpo_log.txt &&

## DPO에서 온전한 모델을 픽스하고, 양자화된 base model이 따로 저장되었으며, 추론에 사용할 프롬프트를 입력
nohup python vllm_inference.py --base_model_path="base_model/Llama-3.1-8B-Instruct-nf4"\
                               --adapter_path="adapter/Zip-Llama-aligned"\
                               --inference_data="data/inference_data.json"\
                               --output_dir="data/inference_result.csv"\
                               --gpu_memory_util=0.45 &&