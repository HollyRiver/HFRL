from datasets import load_dataset

## 원시 데이터 로드
ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

ds_split = ds["train"].train_test_split(test_size = 0.5, seed = 42)

## For SFT
sft_ds = ds_split["train"]
sft_ds = sft_ds.rename_column("chosen", "messages").remove_columns([col for col in sft_ds.column_names if col != "chosen"]).train_test_split(test_size = 0.1, seed = 42)
sft_ds["train"].to_json("./data/sft_train_dataset.json", orient = "records")
sft_ds["test"].to_json("./data/sft_test_dataset.json", orient = "records")

## Implicit Prompt -> Explicit Prompt
dpo_ds = ds_split["test"].map(
    lambda sample: {
        "prompt": [{"role": "user", "content": sample["prompt"]}],
        "chosen": [content for content in sample["chosen"] if content["role"] == "assistant"],
        "rejected": [content for content in sample["rejected"] if content["role"] == "assistant"]
    }
)

dpo_ds = dpo_ds.remove_columns([col for col in dpo_ds.column_names if col not in ["prompt", "chosen", "rejected"]]).train_test_split(test_size = 0.1, seed = 42)
dpo_ds["train"].to_json("./data/dpo_train_dataset.json", orient = "records")
dpo_ds["test"].to_json("./data/dpo_test_dataset.json", orient = "records")