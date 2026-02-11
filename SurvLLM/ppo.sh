## 선호도 데이터셋이 준비되었을 때 사용
nohup python RM.py --config config/RM_config_v1.0.0.yaml > logs/rm_log_v1.0.0.txt &

## 단순 입력 프롬프트만 존재하는 데이터 사용
nohup python PPO.py --config config/PPO_config_v1.0.0.yaml > logs/ppo_log_v1.0.0.txt &