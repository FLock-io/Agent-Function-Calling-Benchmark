import json
import os
import re
from tqdm import tqdm
import ast


def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True  # 如果没有抛出异常，说明字符串是有效的JSON
    except json.JSONDecodeError:
        return False  # 如果抛出异常，说明字符串不是有效的JSON


def recursive_json_parse(obj):
    # 如果对象是字符串，尝试解析它
    if isinstance(obj, str):
        try:
            # 尝试将字符串解析为JSON
            return json.loads(obj)
        except json.JSONDecodeError as e:
            # 如果不是有效的JSON字符串，返回原始字符串
            print(e)
            return obj
    # 如果对象是字典或列表，递归处理每个元素
    elif isinstance(obj, dict):
        return {key: recursive_json_parse(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [recursive_json_parse(item) for item in obj]
    # 对于其他类型的对象，直接返回
    return obj


def prepare_eval_data():
    data_path = "./bfcl_v3_simple/lr_exp_test_generated_predictions.jsonl"
    all_data = open(data_path, "r").readlines()
    eval_data = []
    user_pattern = r"<im_start>user\n(.*?)<im_end>\n"
    for line in tqdm(all_data):
        data = json.loads(line)
        query = data["prompt"]
        predict = data["predict"]
        answer = data["label"]
        # query = re.search(user_pattern, prompt, re.DOTALL)
        # query = query.group(1) if query else ""
        # query = data["query"]
        # predict = predict.replace(r"\"", '"')

        if is_valid_json(predict):
            predict = [recursive_json_parse(pred) for pred in json.loads(predict)]
        else:
            predict = []
        if is_valid_json(answer):
            answer = [recursive_json_parse(ans) for ans in json.loads(answer)]
            # format bfcl expected tool
            formatted_answers = []
            for item in answer:
                arguments = item["arguments"]
                formatted_arguments = {}
                for key, value in arguments.items():
                    if isinstance(value, str):
                        formatted_arguments[key] = value
                    formatted_arguments[key] = value
                    if isinstance(value, list) & len(value) == 1:
                        formatted_arguments[key] = value[0]
                formatted_answers.append(
                    {"name": item["name"], "arguments": formatted_arguments}
                )
            answer = formatted_answers
        else:
            answer = []
        eval_data.append(
            {"query": query, "gold_tools": answer, "predict_tools": predict}
        )
    with open("./bfcl_v3_simple/lr_exp_test_set_evaluation_data.jsonl", "w") as f:
        for tool in eval_data:
            f.write(json.dumps(tool, ensure_ascii=False) + "\n")


def prepare_baseline_data():
    data_path = "./bfcl_v3_parallel_multi/BFCL_v3_parallel_multi_test_data.json"
    full_data = json.load(open(data_path, "r"))
    input_data = []
    for data in full_data:
        conversations = data["conversations"]
        for conv in conversations:
            if conv["role"] == "user":
                query = conv["content"]
            if conv["role"] == "assistant":
                answer = conv["content"]
        system = data["system"]
        available_tools = system.split("Use them if required - ")[-1]
        available_tools = json.loads(available_tools)
        formated_tools = []
        for tool in available_tools:
            formated_tools.append({"type": "function", "function": tool})
        # answers = json.loads(answer)
        answers = [
            json.loads(ans) if isinstance(ans, str) else ans
            for ans in json.loads(answer)
        ]
        # format bfcl expected tool
        formatted_answers = []
        for item in answers:
            arguments = item["arguments"]
            formatted_arguments = {}
            for key, value in arguments.items():
                if isinstance(value, str):
                    formatted_arguments[key] = value
                formatted_arguments[key] = value
                if isinstance(value, list) & len(value) == 1:
                    formatted_arguments[key] = value[0]
            formatted_answers.append(
                {"name": item["name"], "arguments": formatted_arguments}
            )
        input_data.append(
            {"query": query, "answers": formatted_answers, "tools": formated_tools}
        )

    with open("./bfcl_v3_parallel_multi/baseline_data.jsonl", "w") as f:
        for tool in input_data:
            f.write(json.dumps(tool, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    prepare_eval_data()
    # prepare_baseline_data()
