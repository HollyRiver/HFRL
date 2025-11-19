import pandas as pd
import datasets
import argparse
from utils import remove_hangul

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type = str, default = None, help = "변환할 파일 이름")
    parser.add_argument("--encoding", type = str, default = "utf-8", help = "변환할 파일 인코딩")

    args = parser.parse_args()

    ## 원시 데이터 로드
    df_text = pd.read_csv(f"data/{args.target}", encoding = args.encoding)
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
    train_ds = train_ds.map(lambda sample: remove_hangul(sample, column = "messages"))

    train_ds.to_json(f"data/{args.target.split(".")[0]}.json", orient = "records")