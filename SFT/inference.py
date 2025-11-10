## 그냥 콜백에 추가해서 모델 훈련 시 에폭마다 자동으로 추론 결과를 산출-저장하도록 만드는 게 제일 좋을 듯?
import argparse
import os
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "For model setting")
    parser.add_argument("--model_path", type = str, default = None, help = "Inference LLM model folder path")
    parser.add_argument("--temp", type = float, default = 0.4, help = "Generating temperature")
    config = parser.parse_args()

    test_dataset = load_dataset("json", data_files = os.path.join("", "./test_dataset.json"), split = "train")

    ## 파인튜닝되지 않은 모델의 토크나이저에 적용시켜줄 것.
    LLAMA_3_CHAT_TEMPLATE = (
        "{{ bos_token }}"
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + eos_token }}"
            "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] +  eos_token }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n'}}"
                "{% generation %}"
                "{{ message['content'] +  eos_token }}"
                "{% endgeneration %}"
            "{% endif %}"
        "{% endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{%- endif %}"
    )

    model_name = config.model_path

    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache = False, device_map = "cuda:0", dtype = torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    result = []

    for idx in tqdm(range(test_dataset.num_rows)):
        messages = test_dataset[idx]["messages"][:2]

        terminators = [
            tokenizer.eos_token_id,
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt"           ## convert to pytorch tensor
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens = 512,
            eos_token_id = terminators,
            do_sample = True,
            temperature = config.temp,
            top_p = 0.95,
            pad_token_id = tokenizer.eos_token_id
        )

        response = outputs[0][input_ids.shape[-1]:]
        question = test_dataset[idx]["messages"][1]["content"]
        answer = test_dataset[idx]["messages"][2]["content"]
        generation = tokenizer.decode(response, skip_special_tokens = True)
        result.append([question, answer, generation])

    os.mkdir("inference")

    with open("./inference/model_generation_result.txt", "w") as f:
        for line in result:
            f.write(str(line) + "\n")