import pandas as pd
import datasets

## 원시 데이터 로드
df_text = pd.read_csv("data/data_sample_20251111.csv", encoding = "cp949")
ds = datasets.Dataset.from_pandas(df_text)
columns_to_remove = [f for f in list(ds.features) if f != "subject_id"]

system_prompt = df_text.system[0]
question = df_text.question[0]
# system_prompt = "You are the world’s leading expert in survival analysis.\
#  From a discharge summary, extract Chief Complaint, Physical Exam, and Admission Labs and produce one sentence.\
#  The sentence will be used for hazard calculation, so be precise, clinically accurate, and concise."
# question = "Please summarize the following discharge summary\
#  in one sentence focusing on Chief Complaint, Physical Exam, and Admission Labs."

train_ds = ds.map(
    lambda sample:
    {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["question"] + "\n" + sample["text"]},
        {"role": "assistant", "content": sample["assistant"]}
    ]}
)

train_ds = train_ds.map(remove_columns = columns_to_remove, batched = False)
train_ds = train_ds.train_test_split(test_size = 0.1, seed = 42)

train_ds["train"].to_json("data/sft_train_dataset.json", orient = "records", force_ascii = False)
train_ds["test"].to_json("data/sft_test_dataset.json", orient = "records", force_ascii = False)

test_ds = train_ds["test"]
lst = []

for idx in range(test_ds.num_rows):
    lst.append({"subject_id": test_ds["subject_id"][idx], "label": test_ds["messages"][idx][2]["content"]})

pd.DataFrame(lst).to_csv("data/test_label.csv", index = False)