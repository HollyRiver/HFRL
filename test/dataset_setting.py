from datasets import load_dataset

## 원시 데이터 로드
ds = load_dataset("beomi/KoAlpaca-v1.1a")
columns_to_remove = list(ds["train"].features)  ## 전처리 이후 제거할 기존 피쳐

system_prompt = "당신은 다양한 분야의 전문가들이 제공한 지식과 정보를 바탕으로 만들어진 AI 어시스턴트입니다.\
 사용자들의 질문에 대해 정확하고 유용한 답변을 제공하는 것이 당신의 주요 목표입니다. 복잡한 주제에 대해서도\
 이해하기 쉽게 설명할 수 있으며, 필요한 경우 추가 정보나 관련 예시를 제공할 수 있습니다. 항상 객관적이고 중립적인\
 입장을 유지하면서, 최신 정보를 반영하여 답변해 주세요. 사용자의 질문이 불분명한 경우 추가 설명을 요청하고, 당신이\
 확실하지 않은 정보에 대해서는 솔직히 모른다고 말해주세요."

## 전처리 후 저장
train_ds = ds.map(
    lambda sample:
    {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]}
    ]}
)

train_ds = train_ds.map(remove_columns = columns_to_remove, batched = False)
train_ds = train_ds["train"].train_test_split(test_size = 0.1, seed = 42)

train_ds["train"].to_json("train_dataset.json", orient = "records", force_ascii = False)
train_ds["test"].to_json("test_dataset.json", orient = "records", force_ascii = False)