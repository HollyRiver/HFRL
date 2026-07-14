# nohup python preference_AIF.py --model_name="Qwen/Qwen3-30B-A3B" \
#                                --preference_name="data/generated_data_v1.2.0.csv" \
#                                --discharge_name="data/dpo_prompt_data.csv" &

"""
vLLM 판정 모델(AI Feedback)로 5개 생성문의 상대 순위를 매겨 DPO 선호도 데이터셋을 생성하는 코드

    1. long format 생성 데이터(subject_id 당 5행)를 wide format으로 변환
    2. 고정 시스템 프롬프트(파일) + 유저 프롬프트(퇴원요약지 원문 + 5개 라벨)로 순위 JSON 요청
    3. 추론(reasoning) 토큰 이후 출력에서 순위 JSON을 추출,
       순위 1(최고)을 chosen, 순위 5(최저)를 rejected로 하여 csv 저장

시스템 프롬프트는 txt 파일로 저장되어 입력됩니다. 프롬프트를 변경하고 싶으면 해당 파일을 수정하세요.
시스템 프롬프트는 모든 요청에 동일하게 적용되므로 vLLM prefix caching의 이점을 얻습니다.
"""

## 최신 vllm 버전 및 리눅스 24.04 이상 권장
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import pandas as pd
import numpy as np

import torch
import json
import re
import argparse

N_GENERATIONS = 5                       ## subject_id 당 생성문 개수
GEN_COLS = list(range(1, N_GENERATIONS + 1))
MAX_MODEL_LEN = 32768
MAX_NEW_TOKENS = 512                    ## 추론(reasoning) 토큰 + 순위 JSON

## 유저 프롬프트 템플릿. 변경할 일이 없으므로 별도 파일 대신 코드에 내장
USER_PROMPT_TEMPLATE = """Below is the ORIGINAL discharge summary of a patient. This is the ground-truth clinical record.

===== ORIGINAL DISCHARGE SUMMARY =====
{original_text}
======================================

Here are the 5 generated clinical summaries (Labels) to evaluate.

[Label 1]
{label_1_text}

[Label 2]
{label_2_text}

[Label 3]
{label_3_text}

[Label 4]
{label_4_text}

[Label 5]
{label_5_text}

Critically evaluate these labels based on the provided system priorities. Output ONLY the final valid JSON object mapping the label indices to their unique relative ranks. Do not include any other text."""


def build_user_content(original_text: str, generations: list[str]) -> str:
    """퇴원요약지 원문과 5개 생성문을 유저 프롬프트 템플릿에 채워 넣는다."""
    labels = {f"label_{i}_text": text for i, text in enumerate(generations, start = 1)}
    return USER_PROMPT_TEMPLATE.format(original_text = original_text, **labels)


def parse_rank_label(json_text: str) -> dict[str, int] | None:
    """JSON 문자열을 순위 dict로 파싱. 유효하지 않으면 None 반환."""
    try:
        label = json.loads(json_text.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(label, dict) or set(label.keys()) != {str(i) for i in GEN_COLS}:
        return None

    values = list(label.values())

    ## JSON true/false는 파이썬에서 int의 하위 타입(bool)이므로 명시적으로 거른다
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return None
    ## 허용 범위(1~5)를 벗어난 순위는 제외
    if min(values) < 1 or max(values) > N_GENERATIONS:
        return None
    ## 모든 순위가 동일하면 선호 비교가 불가능하므로 스킵 (예: {3,3,3,3,3})
    if len(set(values)) == 1:
        return None

    return label


def extract_rank_label(raw_text: str) -> dict[str, int] | None:
    """추론(reasoning) 토큰을 포함한 모델 출력에서 순위 JSON을 추출. 실패 시 None 반환."""
    ## 추론이 max_tokens에 잘려 </think>가 닫히지 않은 경우, 추론 중 되뇌인 예시 JSON을
    ## 실제 순위로 오인할 수 있으므로 무효 처리 (추론 없는 모델의 출력은 그대로 탐색)
    if "<think>" in raw_text and "</think>" not in raw_text:
        return None

    ## 추론 과정에 예시 JSON이 섞일 수 있으므로 </think> 이후 텍스트만 사용
    answer_text = raw_text.split("</think>")[-1]

    ## 중첩 없는 JSON 객체 후보를 모두 찾아 마지막 유효 후보를 채택
    candidates = re.findall(r"\{[^{}]*\}", answer_text)

    for candidate in reversed(candidates):
        label = parse_rank_label(candidate)

        if label is not None:
            return label

    return None


def build_wide_generations(df_gen: pd.DataFrame) -> pd.DataFrame:
    """long format 생성 데이터를 subject_id 당 생성문 5개를 열(1~5)로 갖는 wide format으로 변환."""
    ## 행 순서가 subject_id 당 정확히 5행 단위라는 가정 대신 groupby로 안전하게 생성 번호 부여
    counts = df_gen.groupby("subject_id").size()
    invalid_ids = counts.index[counts != N_GENERATIONS]

    if len(invalid_ids) > 0:
        print(f"[Warning] 생성문이 {N_GENERATIONS}개가 아닌 subject_id {len(invalid_ids)}건 제외: {list(invalid_ids[:10])} ...")

    df_valid = df_gen.loc[df_gen["subject_id"].isin(counts.index[counts == N_GENERATIONS])]
    df_valid = df_valid.assign(gen_num = df_valid.groupby("subject_id").cumcount() + 1)

    return df_valid.pivot(index = "subject_id", columns = "gen_num", values = "generated_text").reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, default = "Qwen/Qwen3-30B-A3B", help = "Inference Model")
    parser.add_argument("--preference_name", type = str, default = None, help = "Preference Dataset Name (Long Format)")
    parser.add_argument("--discharge_name", type = str, default = "dpo_prompt_data.csv", help = "동일한 인덱스를 가지는 퇴원요약지 데이터셋")
    parser.add_argument("--system_prompt", type = str, default = "preference_system_prompt.txt", help = "평가 기준 시스템 프롬프트 txt 파일 위치")
    parser.add_argument("--output_name", type = str, default = None, help = "생성될 DPO 데이터셋 저장 위치")

    args = parser.parse_args()

    if args.preference_name is None or args.output_name is None:
        parser.error("--preference_name과 --output_name은 필수 인자입니다.")

    ## ---------- 데이터 준비 (모델 로드 전에 수행하여 입력 문제 시 빠르게 실패) ----------
    with open(args.system_prompt, "r", encoding = "utf-8") as f:
        system_prompt = f.read().strip()

    df_gen = pd.read_csv(args.preference_name)
    df_discharge = pd.read_csv(args.discharge_name)

    if not {"subject_id", "generated_text"}.issubset(df_gen.columns):
        raise ValueError(f"{args.preference_name}에 subject_id, generated_text 열이 필요합니다.")
    if not {"subject_id", "text"}.issubset(df_discharge.columns):
        raise ValueError(f"{args.discharge_name}에 subject_id, text 열이 필요합니다.")

    ## 두 CSV의 subject_id dtype이 다르면(int64 vs object) merge가 조용히 빈 결과를 내므로 str로 통일
    df_gen = df_gen.assign(subject_id = df_gen["subject_id"].astype(str))
    df_discharge = df_discharge.assign(subject_id = df_discharge["subject_id"].astype(str))

    df_discharge = df_discharge.drop_duplicates(subset = "subject_id")
    df_wide = build_wide_generations(df_gen)

    ## 5개 생성문이 모두 동일한 subject_id는 LLM 평가 자체를 건너뜀 (선호도 산출 불가)
    df_wide = df_wide.loc[df_wide[GEN_COLS].nunique(axis = 1) > 1]

    ## 퇴원요약지 원문 병합 (원문이 없는 subject_id는 제외)
    df_eval = pd.merge(df_wide, df_discharge[["subject_id", "text"]], on = "subject_id", how = "inner").reset_index(drop = True)

    if len(df_eval) < len(df_wide):
        print(f"[Warning] 퇴원요약지 원문이 없는 subject_id {len(df_wide) - len(df_eval)}건 제외")

    if len(df_eval) == 0:
        raise SystemExit("[Error] 평가할 데이터가 없습니다. 입력 파일을 확인하세요.")

    ## ---------- 프롬프트 구성 ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast = True)

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_content(text, generations)}
            ],
            tokenize = False, add_generation_prompt = True
        )
        for text, generations in zip(df_eval["text"], df_eval[GEN_COLS].to_numpy().tolist())
    ]

    ## 컨텍스트 길이 초과 프롬프트가 하나라도 있으면 vLLM 실행 전체가 실패하므로 사전에 제외
    prompt_lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens = False)["input_ids"]]
    keep_mask = [(length + MAX_NEW_TOKENS) <= MAX_MODEL_LEN for length in prompt_lengths]

    if not all(keep_mask):
        dropped_ids = df_eval.loc[[not keep for keep in keep_mask], "subject_id"].tolist()
        print(f"[Warning] 컨텍스트 길이({MAX_MODEL_LEN}) 초과로 subject_id {len(dropped_ids)}건 제외: {dropped_ids[:10]} ...")
        df_eval = df_eval.loc[keep_mask].reset_index(drop = True)
        prompts = [prompt for prompt, keep in zip(prompts, keep_mask) if keep]

    ## ---------- LLM 추론 ----------
    llm = LLM(
        model = args.model_name,
        dtype = torch.bfloat16,
        trust_remote_code = True,
        max_model_len = MAX_MODEL_LEN,
        gpu_memory_utilization = 0.9
    )

    ## 추론(reasoning)을 허용하기 위해 출력 형태 제한(structured outputs)을 두지 않음
    sampling_params = SamplingParams(temperature = 0.0, max_tokens = MAX_NEW_TOKENS)

    outputs = llm.generate(prompts, sampling_params)

    ## ---------- 순위 파싱 및 DPO 데이터셋 구성 ----------
    subject_ids = df_eval["subject_id"].tolist()
    texts = df_eval["text"].tolist()
    gen_matrix = df_eval[GEN_COLS].to_numpy()

    data = []
    n_invalid_label = 0
    n_truncated = 0

    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text
        label = extract_rank_label(raw_text)

        if label is None:
            n_invalid_label += 1

            ## 추론이 잘려 무효 처리된 건수는 별도 집계 (MAX_NEW_TOKENS 조정 판단용)
            if "<think>" in raw_text and "</think>" not in raw_text:
                n_truncated += 1

            continue

        ## 순위 1 = 가장 우수(chosen), 5 = 가장 열등(rejected)
        ## 순위 동률이면 argmin/argmax는 가장 앞 위치를 반환
        ranks = [label[str(k)] for k in GEN_COLS]
        chosen = gen_matrix[i][int(np.argmin(ranks))]
        rejected = gen_matrix[i][int(np.argmax(ranks))]

        ## 동일 텍스트 쌍은 선호 신호가 없으므로 제외 (중복 생성문에 서로 다른 순위가 매겨진 경우)
        if chosen == rejected:
            continue

        data.append({
            "subject_id": subject_ids[i],
            "text": texts[i],
            "chosen": chosen,
            "rejected": rejected
        })

    if len(data) == 0:
        raise SystemExit("[Error] 유효한 선호도 쌍이 생성되지 않았습니다. 판정 모델 출력을 확인하세요.")

    dpo_dataset = pd.DataFrame(data)
    dpo_dataset.to_csv(args.output_name, index = False, encoding = "utf-8-sig")

    print(f"\n평가 대상 {len(df_eval)}건 중 유효하지 않은 라벨 {n_invalid_label}건(추론 잘림 {n_truncated}건 포함), 동일 텍스트 쌍 {len(df_eval) - n_invalid_label - len(data)}건 제외")
    print(f"DPO 데이터셋 {len(dpo_dataset)}건 저장 완료: {args.output_name}")
