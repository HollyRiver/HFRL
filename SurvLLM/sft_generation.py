import os
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, default = None, help = "model path")
    parser.add_argument("--output_name", type = str, default = "for_dpo_5_gen.csv")
    args = parser.parse_args()

    model_name = args.model_name

    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache = False, device_map = "cuda:0", dtype = torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    gen_ds = load_dataset("json", data_files = "data/dpo_resource.json")["train"]
    output_path = os.path.join("data", f"dpo_dataset_generated.csv")

    results = []

    for idx in tqdm(range(gen_ds.num_rows)):
        ith_inference = {"subject_id" : gen_ds[idx]["subject_id"]}
        ith_inference["text"] = gen_ds[idx]["messages"][1]["content"]

        for i in range(5):
            input_ids = tokenizer.apply_chat_template(
                            gen_ds[idx]["messages"],
                            add_generation_prompt=True,
                            return_tensors="pt"
            ).to(model.device)

            terminators = [tokenizer.eos_token_id]

            outputs = model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=1.0,
                top_p = 0.95
            )

            response = outputs[0][input_ids.shape[-1]:]
            generation = tokenizer.decode(response, skip_special_tokens=True)
            ith_inference[f"Gen_{i}"] = generation

        results.append(ith_inference)

    pd.DaraFrame(results).to_csv(f"data/{args.output_name}")