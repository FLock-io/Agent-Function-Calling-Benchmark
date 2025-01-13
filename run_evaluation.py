import json
from function_call_eval import strict_ordered_eval

def run_evaluation(data_path):
    all_data = open(data_path, "r").readlines()
    exact_acc = 0
    call_by_call_acc = 0
    total_prompts = len(all_data)
    for line in all_data:
        data = json.loads(line)
        expected_tool = data["gold_tools"]
        actual_tool = data["predict_tools"]
        exact_sequence_match_accuracy, call_by_call_accuracy = strict_ordered_eval(expected_tool, actual_tool)
        exact_acc += exact_sequence_match_accuracy
        call_by_call_acc += call_by_call_accuracy
    
    exact_acc = exact_acc / total_prompts
    call_by_call_acc = call_by_call_acc / total_prompts
    return exact_acc, call_by_call_acc

if __name__ == "__main__":
    exact_acc, call_by_call_acc = run_evaluation("results.jsonl")
    print(f"Exact sequence match accuracy: {exact_acc}")
    print(f"Call by call accuracy: {call_by_call_acc}")