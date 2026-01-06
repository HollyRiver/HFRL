import pandas as pd
import datasets
import argparse
from utils import remove_hangul

"""
csv 파일을 json 포맷의 SFT/DPO/Inference dataset으로 변환하기 위한 코드
시스템 프롬프트는 txt 파일로 저장되어 입력됩니다. 프롬프트를 변경하고 싶으면 해당 파일을 수정하세요.

SFT csv dataset input format:
    SFT 작업 수행에 필요한 데이터셋의 csv 포맷

    requirements:
        csv 파일 상에서 필요한 column 이름
        subject_id: 텍스트 인덱스. 해당 부분은 식별을 위한 사항이므로, reset_index를 통해 임의로 설정하거나 코드를 수정하여 제거해도 무방
        text: 시스템 프롬프트 다음에 기재될 Input Text.
        assistant: 모델이 학습할 Output Text. 직접 라벨링하거나(Human Feedback) 기계로 라벨링(AI Feedback)

DPO csv dataset input format:
    DPO 작업 수행에 필요한 데이터셋의 csv 포맷

    requirements:
        subject_id: 텍스트 인덱스
        text: Input Text
        chosen: Input Text에 대한 답변으로써 선호되는 텍스트
        rejected: Input Text에 대한 답변으로써 chosen보다 선호되지 못하는 텍스트

Inference csv dataset input format:
    LLM으로부터 생성 작업을 수행하기 위해 필요한 데이터셋의 csv 포맷

    requirements:
        subject_id: 텍스트 인덱스
        text: Input Text. 해당 텍스트에 대한 답변을 반환하는 것이 목적
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type = str, default = None, help = "변환할 csv 파일 위치")
    parser.add_argument("--encoding", type = str, default = "utf-8", help = "변환할 파일 인코딩")
    parser.add_argument("--system", type = str, default = "data/system_prompt.txt", help = "시스템 프롬프트 기재 txt 파일 위치")

    args = parser.parse_args()

    ## 원시 데이터 로드
    df_text = pd.read_csv(args.target, encoding = args.encoding)

    if "tie" in df_text.columns:
        df_text = df_text.loc[lambda _df: _df.tie == "N"]

    ds = datasets.Dataset.from_pandas(df_text)
    columns_to_remove = [f for f in list(ds.features) if f not in ["subject_id", "chosen", "rejected"]]

    with open(args.system, "r") as f:
        system_prompt = f.read()

    if "assistant" in ds[0].keys():
        print("\n========================================")
        print("\"assistant\" column has found. Start to generate sft dataset.")
        train_ds = ds.map(
            lambda sample:
            {"messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["text"]},
                {"role": "assistant", "content": sample["assistant"]}
            ]}
        )
    
        train_ds = train_ds.map(remove_columns = columns_to_remove, batched = False)
        train_ds = train_ds.map(lambda sample: remove_hangul(sample, column = "messages"))
        train_ds = train_ds.train_test_split(test_size = 0.1, seed = 42)

        train_ds["train"].to_json("data/sft_train_dataset.json", orient = "records")
        train_ds["test"].to_json("data/sft_test_dataset.json", orient = "records")

        test_ds = train_ds["test"]
        lst = []

        for idx in range(test_ds.num_rows):
            lst.append({"subject_id": test_ds["subject_id"][idx], "label": test_ds["messages"][idx][2]["content"], "text": test_ds["messages"][idx][1]["content"]})

        pd.DataFrame(lst).to_csv("data/test_label.csv", index = False, encoding = "utf-8-sig")

        print("\n\nTest Label file was saved.")

    elif "chosen" in ds[0].keys():
        print("========================================")
        print("\n\n\"chosen\" column has found. Start to generate dpo dataset.")
        train_ds = ds.map(
            lambda sample: {
                "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sample["text"]}
                    ],
                "chosen": [{"role": "assistant", "content": sample["chosen"]}],
                "rejected": [{"role": "assistant", "content": sample["rejected"]}]
            }
        )

        train_ds = train_ds.map(lambda sample: remove_hangul(sample, column = "prompt"))
        train_ds = train_ds.map(remove_columns = columns_to_remove, batched = False)
        train_ds = train_ds.train_test_split(test_size = 0.1, seed = 42)

        train_ds["train"].to_json("data/dpo_train_dataset.json", orient = "records")
        train_ds["test"].to_json("data/dpo_test_dataset.json", orient = "records")

        test_ds = train_ds["test"]

        lst = []

        for idx in range(test_ds.num_rows):
            lst.append({"subject_id": test_ds["subject_id"][idx], "chosen": test_ds[idx]["chosen"][0]["content"], "rejected": test_ds[idx]["rejected"][0]["content"], "text": test_ds[idx]["prompt"][1]["content"]})

        pd.DataFrame(lst).to_csv("data/dpo_test_label.csv", index = False, encoding = "utf-8-sig")

        print("\n\nTest Label file was saved.")

    else:
        print("\n==============================")
        print("Label column does not found. Start to generate inference dataset.")
        train_ds = ds.map(
            lambda sample:
            {"messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["text"]}
            ]}
        )

        train_ds = train_ds.map(remove_columns = columns_to_remove, batched = False)
        train_ds = train_ds.map(lambda sample: remove_hangul(sample, column = "messages"))

        train_ds.to_json(f"{args.target.split(".")[0]}.json", orient = "records")

        print("\n\nInference dataset was generated.")