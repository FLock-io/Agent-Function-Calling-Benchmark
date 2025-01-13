import json
from function_call_eval import strict_ordered_eval
from openai import OpenAI
from utils import get_chat_completion
from run_evaluation import run_evaluation

# baseline testing
def run_baseline(model, sys_prompt, data):
    """
    run proprietary models given data
    - model(str): model name
    - sys_prompt(str): system prompt
    - data (list(dict)): 
        query(str): user's query
        answers(str): expected answers
        tools(str): list of predefined functions 
        
    """
    actual_tools = []

    for item in data:
        query = item['query']
        answers = item['answers']
        tools = item['tools']
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]
        completion = get_chat_completion(
            messages=messages,
            model=model,
            tools=tools,
            temperature=0.0,
            tool_choice="required"
        )
        prediction = completion.tool_calls
        pred_tools = [{"name": pred.function.name, "arguments": json.loads(pred.function.arguments)} for pred in prediction]
        actual_tools.append({"query": query, "gold_tools": answers, "predict_tools": pred_tools})
    
    with open(f"baseline_{model}_results.jsonl", "w") as f:
        for tool in actual_tools:
            f.write(json.dumps(tool) + "\n")
    


if __name__ == "__main__":
    DRONE_SYSTEM_PROMPT = """You are an intelligent AI that controls a drone. Given a command or request from the user,
                             call one of your functions to complete the request. If the request cannot be completed by your available functions, call the reject_request function.
                             If the request is ambiguous or unclear, reject the request."""
    example_data = []
    with open("./example_data.jsonl", "r") as f:
        for line in f.readlines():
            example_data.append(json.loads(line))
    
    # run baseline models
    run_baseline("gpt-4o-mini", DRONE_SYSTEM_PROMPT, example_data)

    # you can also only run the evaluation with the provided results
    exact_acc, call_by_call_acc = run_evaluation("baseline_gpt-4o-mini_results.jsonl")
    print(f"Exact sequence match accuracy: {exact_acc}")
    print(f"Call by call accuracy: {call_by_call_acc}")

