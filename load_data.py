from datasets import load_dataset
import json
import re
import random

def extract_data(input_text):
    # Regular expressions to extract each part
    available_tools_pattern = r"<s>\[AVAILABLE_TOOLS\](.*?)\[/AVAILABLE_TOOLS\]"
    inst_pattern = r"\[INST\](.*?)\[/INST\]"
    tool_calls_pattern = r"\[TOOL_CALLS\](.*?)</s>"

    # Extract matches
    available_tools_match = re.search(available_tools_pattern, input_text, re.DOTALL)
    inst_match = re.search(inst_pattern, input_text, re.DOTALL)
    tool_calls_match = re.search(tool_calls_pattern, input_text, re.DOTALL)

    available_tools = json.loads(available_tools_match.group(1)) if available_tools_match else None
    inst = inst_match.group(1).strip() if inst_match else None
    tool_calls = json.loads(tool_calls_match.group(1)) if tool_calls_match else None

    return available_tools, inst, tool_calls

datasets = load_dataset("vietgpt/glaive-function-calling-v2")


output_file = "./example_data.jsonl"
all_data = []
for data in datasets["train"]:
    available_tools, inst, tool_calls = extract_data(data["text"])

    if tool_calls is not None:
        all_data.append({"query": inst, "answers": tool_calls, "tools": available_tools})

with open(output_file, "w") as f:
    for data in random.sample(all_data, 100):
        f.write(json.dumps(data) + "\n")