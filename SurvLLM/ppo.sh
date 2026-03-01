## 선호도 데이터셋이 준비되었을 때 사용
nohup python RM.py --config config/RM_config_v1.1.2.3.yaml > logs/rm_log_v1.1.2.3.txt &

## ppo 데이터셋 준비
nohup python csv_to_json_dataset.py --target="data/ppo_dataset.csv"\
                                    --encoding="utf-8"\
                                    --system="data/system_prompt.txt"\
                                    --ppo=true &

## 단순 입력 프롬프트만 존재하는 데이터 사용
nohup python PPO.py --config config/PPO_config_v1.1.2.0.yaml > logs/ppo_log_v1.1.2.0.txt &