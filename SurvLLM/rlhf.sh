## 여기 있는거 한번에 다 안돼요. 성능상 한번에 해도 안되고.
## 그리고 실행할 때에는 &&가 아닌 &를 붙여야 합니다. 여긴 그냥 순차적으로 된다는 가정하에 작성했어요.

nohup python csv_to_json_dataset.py --target="data/data_sample_20251111_01.csv"\
                                    --encoding="cp949"\
                                    --system="data/system_prompt.txt" &&

nohup python SFT.py --config config/SFT_config.yaml > sft_log.txt &&

## SFT에서 온전한 모델을 픽스하고, 해당 어뎁터를 삽입
## temperature 설정은 1.0 정도로 해야 다양한 결과 나옴
nohup python sft_generate.py --adapter_name="adapter/Zip-Llama-sft"\
                             --output_name="gen_data.csv"\
                             --gen_nums=5\
                             --temp=1.0 &&

## SFT에서 온전한 모델을 픽스하고, 데이터셋이 준비되었을 때
nohup python csv_to_json_dataset.py --target="data/gen_data_20251118_for_dpo.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt" &&

nohup python DPO.py --config config/DPO_config.yaml > dpo_log.txt &&

nohup python gen_llama_nf4.py &&

nohup python csv_to_json_dataset.py --target="data/data_all.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt" &&

## DPO에서 온전한 모델을 픽스하고, 양자화된 base model이 따로 저장되었으며, 추론에 사용할 프롬프트가 준비되었을 때
nohup python vllm_inference.py --base_model_path="base_model/Llama-3.1-8B-Instruct-nf4"\
                               --adapter_path="adapter/Zip-Llama-aligned"\
                               --inference_data="data/data_all.json"\
                               --output_dir="data/inference_all_greedy_with_penalty.csv"\
                               --sampling=False\
                               --repetition_penalty=1.1\
                               --gpu_memory_util=0.45 &&