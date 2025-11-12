import pandas as pd
import datasets
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target", type = str, default = None, help = "변환할 파일 이름")

args = parser.parse_args()

def remove_hangul_from_messages(sample):
    hangul_pattern = r"[\uAC00-\uD7A3]"

    cleaned_messages = []
    for message in sample["messages"]:
        # content 값에서 한글을 제거
        cleaned_content = re.sub(hangul_pattern, "", message["content"])
        
        # 정제된 content로 메시지 딕셔너리 재생성
        cleaned_messages.append({
            "role": message["role"],
            "content": cleaned_content
        })
    
    # 정제된 messages 리스트를 샘플에 다시 할당
    sample["messages"] = cleaned_messages
    return sample

## 원시 데이터 로드
df_text = pd.read_csv(f"data/{args.target}", encoding = "cp949")
ds = datasets.Dataset.from_pandas(df_text)
columns_to_remove = [f for f in list(ds.features) if f != "subject_id"]

system_prompt = pd.read_csv("data/data_sample_20251111_01.csv", encoding = "cp949").system[0]

train_ds = ds.map(
    lambda sample:
    {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["text"]},
        {"role": "assistant", "content": sample["assistant"]}
    ]} if "assistant" in sample.keys() else
    {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["text"]}
    ]}
)

train_ds = train_ds.map(remove_columns = columns_to_remove, batched = False)
train_ds = train_ds.map(remove_hangul_from_messages)

train_ds.to_json(f"data/{args.target.split(".")[0]}.json", orient = "records")