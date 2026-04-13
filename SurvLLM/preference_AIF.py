## 최신 vllm 버전 및 리눅스 24.04 이상 권장
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer
from datasets import Dataset

import pandas as pd
import numpy as np

import torch
import json
import argparse

def build_system_content(original_text):
    return (
        "You are an expert medical evaluator.\n\n"
        "Below is the ORIGINAL discharge summary of a patient.\n"
        "This is the ground-truth clinical record.\n\n"
        "===== ORIGINAL DISCHARGE SUMMARY =====\n"
        f"{original_text}\n"
        "======================================\n\n"
        "You will be given 5 generated clinical summaries of the SAME patient.\n"
        "Evaluate each summary independently, but assign scores based on RELATIVE quality,\n"
        "following the priority order specified below.\n\n"
        "EVALUATION PRIORITIES (VERY IMPORTANT):\n\n"
        "Priority 1 (MOST IMPORTANT): Correct use of REQUIRED TEST NAMES and SEX\n"
        "- The following test names must be explicitly and correctly written:\n"
        "  - Vitals: BP, HR, RR, Temp\n"
        "  - Labs: WBC, RBC, Hgb, Hct, Plt\n"
        "  - Red cell indices and others: MCV, MCH, MCHC, RDW, Glucose\n"
        "- Patient sex (Male/Female) must be correctly stated\n"
        "- Incorrect, missing, or improperly named tests or sex are penalized\n"
        "- This priority ALWAYS outweighs all lower priorities\n\n"
        "Priority 2: Correctness of VALUES corresponding to each test name\n"
        "- When a value is explicitly stated, it must match the discharge summary exactly\n"
        "- If a test is NOT present in the discharge summary, NA / not available / not reported\n"
        "  is considered CORRECT\n"
        "- Hallucinated, incorrect, or contradictory values are penalized\n\n"
        "Priority 3: Appropriateness of the clinical finding sentence\n"
        "- The sentence following \"The most diagnostic relevant finding was ...\" should:\n"
        "  - Describe a clinically meaningful finding or observation\n"
        "  - NOT be merely a diagnosis name\n"
        "  - Be consistent with the discharge summary\n\n"
        "SCORING RULES:\n"
        "- Scores represent RELATIVE ranks among the 5 summaries (5 = worst, 1 = best)\n"
        "- Multiple summaries may share the same score if their quality is comparable\n"
        "- Even if all summaries are poor, you MUST still rank them relative to each other\n\n"
        "OUTPUT RULES (STRICT):\n"
        "- You MUST output ONLY a single-line valid JSON object\n"
        "- The JSON must map summary index to score\n"
        "- Any token outside the JSON object is STRICTLY FORBIDDEN\n"
        "- Do NOT provide explanations, reasoning, comments, or natural language\n"
        "- Do NOT use markdown or headings\n\n"
        "Example:\n"
        "{\"1\": 1, \"2\": 3, \"3\": 2, \"4\": 4, \"5\": 5}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, default = "Qwen/Qwen3-30B-A3B", help = "Inference Model")
    parser.add_argument("--preference_name", type = str, default = None, help = "Preference Dataset Name (Long Format)")
    parser.add_argument("--discharge_name", type = str, default = None, help = "동일한 인덱스를 가지는 퇴원요약지 데이터셋")

    args = parser.parse_args()

    llm = LLM(
        model = args.model_name,
        dtype = torch.bfloat16,
        trust_remote_code = True,
        max_model_len = 32768,
        gpu_memory_utilization = 0.9
    )

    df_gen = pd.read_csv(args.preference_name)
    df_discharge = pd.read_csv(args.discharge_name)

    summarized_text = df_gen.pivot_table(index = "subject_id", values = "generated_text",
                                        aggfunc = (lambda x: "\n\n".join([f"[{i+1}{"st" if i == 0 else "nd" if i == 1 else "rd" if i == 2 else "th"} generated text]\n\"\"\"\n{t}\n\"\"\"" for i, t in enumerate(x)])))\
                                            .reset_index()
    df_gen = df_gen.assign(gen_num = [(i%5)+1 for i in range(df_gen.shape[0])])
    df_wide = df_gen.pivot(index = "subject_id", columns = "gen_num", values = "generated_text").reset_index()
    full_text = pd.merge(summarized_text, df_discharge[["subject_id", "text"]]).assign(system = lambda _df: _df.text.map(build_system_content)).drop(["text"], axis = 1)

    ds = Dataset.from_pandas(full_text)
    columns_to_remove = [f for f in list(ds.features) if f not in "subject_id"]

    ds = ds.map(
        lambda sample:
        {"messages": [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["generated_text"]}
        ]}
    )

    sampling_params = SamplingParams(temperature = 0.0, max_tokens = 64, structured_outputs=StructuredOutputsParams(json={"type": "object"}))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast = True)

    def template_dataset(example):
        return {"prompt": tokenizer.apply_chat_template(example["messages"], tokenize = False, add_generation_prompt = True)}

    inference_data = ds.map(template_dataset, remove_columns = ["messages"])
    prompts = inference_data["prompt"]

    outputs = llm.generate(prompts, sampling_params)

    data = []
    idx = inference_data["subject_id"]

    for i, output in enumerate(outputs):
        current_subject_id = idx[i]

        try:
            label = json.loads(output.outputs[0].text.strip())
        except:
            label = pd.NA

        row = {
            "subject_id": current_subject_id,
            "label": label
        }
        
        data.append(row)

    df = pd.DataFrame(data)

    data = []

    for id, label in df.itertuples(index = False):
        if set(label.keys()) == {"1", "2", "3", "4", "5"} and max(label.values()) == 5 and min(label.values()) == 1:
            ## 순위의 중복이 있을 경우 제일 먼저 것만 선택
            max_idx = np.argmax(list(label)) + 1
            min_idx = np.argmin(list(label)) + 1
        else:
            continue

        row = {
            "subject_id": id,
            "text": df_discharge.loc[df_discharge.subject_id == id, "text"].item(),
            "chosen": df_wide.loc[df_wide.subject_id == id, min_idx].item(),
            "rejected": df_wide.loc[df_wide.subject_id == id, max_idx].item()
        }

        data.append(row)

    dpo_dataset = pd.DataFrame(data).loc[(lambda _df: _df.chosen != _df.rejected)]
    # dpo_dataset.to_csv("data/dpo_dataset.csv", index = False, encoding = "utf-8-sig")