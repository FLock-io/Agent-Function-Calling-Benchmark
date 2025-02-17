import json
from function_call_eval import strict_ordered_eval
from utils import get_chat_completion
from run_evaluation import run_evaluation
import time
import backoff


@backoff.on_exception(backoff.expo, Exception, max_time=200, max_tries=10)
def completion_with_backoff(**kwargs):
    return get_chat_completion(**kwargs)


# baseline testing
def run_chat(model, data, results_file=None):
    """
    run proprietary models given data
    - model(str): model name
    - sys_prompt(str): system prompt
    - data (list(dict)):
        query(str): user's query
        answers(str): expected answers
        tools(str): list of predefined functions
        results_file(str): file to save the results

    """
    actual_tools = []

    for item in data:
        query = item["query"]
        answers = item["answers"]
        tools = item["tools"]
        output_format = [
            {
                "name": "function_name",
                "arguments": {"arg_name": "arg_value", "arg_name": "arg_value"},
            },
            {
                "name": "function_name",
                "arguments": {"arg_name": "arg_value", "arg_name": "arg_value"},
            },
        ]
        output_prompt = (
            "\n\n"
            + "You must answer with the string format of the following example without any other details: \n\n"
            + json.dumps(output_format, ensure_ascii=False)
            + "\n\n"
        )
        sys_prompt = (
            "You are a helpful assistant with access to the following functions. Use them if required - "
            + json.dumps(tools, ensure_ascii=False)
            + output_prompt
        )
        messages = [
            # {"role": "system", "content": sys_prompt},
            {"role": "user", "content": sys_prompt + "\n\n" + query},
        ]
        try:
            completion = completion_with_backoff(
                messages=messages,
                model=model,
                tools=None,
                temperature=0.0,
                tool_choice=None,
            )
            pred_tools = completion.content
            print(pred_tools)
            pred_tools = json.loads(pred_tools)
            print(f"Prediction: {pred_tools}")
            actual_tools.append(
                {"query": query, "gold_tools": answers, "predict_tools": pred_tools}
            )
        except Exception as e:
            print(f"Error: {e}")
            actual_tools.append(
                {"query": query, "gold_tools": answers, "predict_tools": ""}
            )
            continue

    with open(results_file, "w") as f:
        for tool in actual_tools:
            f.write(json.dumps(tool) + "\n")


if __name__ == "__main__":
    DRONE_SYSTEM_PROMPT = """You are an intelligent AI that controls a drone. Given a command or request from the user,
                             call the functions to complete the request. If the request cannot be completed by your available functions, call the reject_request function.
                             If the request is ambiguous or unclear, reject the request."""
    example_data = []
    with open("./data/bfcl_v3_parallel_multi/baseline_data.jsonl", "r") as f:
        for line in f.readlines():
            example_data.append(json.loads(line))

    model_name = "deepseek-chat"

    results_file = (
        "./data/bfcl_v3_parallel_multi/baseline_{model_name}_chat_results.jsonl".format(
            model_name=model_name
        )
    )
    # run baseline models
    run_chat(model_name, example_data, results_file)

    # you can also only run the evaluation with the provided results
    exact_acc, call_by_call_acc = run_evaluation(results_file)
    print(f"Exact sequence match accuracy: {exact_acc}")
    print(f"Call by call accuracy: {call_by_call_acc}")
