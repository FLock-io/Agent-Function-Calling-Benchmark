import json
import re
import string

# Function to normalize the answer, perhaps we can use this to normalize the function names if needed
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def strict_ordered_eval(expected_tool, actual_tool):
    """
    Evaluate the accuracy of function calls per prompt against a gold-standard list of functions, assuming a strict sequence match.
    
    Args:
        expected_tools (list(dict)): A list of expected functions for the prompt.
            - name: function name
            - arguments (dict): 
        actual_tools (list(dict)): A list of predicted functions for the prompt.

    Returns:
        - exact sequence match accuracy: The proportion of prompts where the entire sequence of calls is correct in name, parameters, and order
        - call by call accuracy: for each prompt, compute how many function calls are correct out of the total expected calls
        For instance, if gold has 2 calls, but the model only got 1 correct (in the correct order with correct parameters), the call-by-call accuracy for that prompt is 50%.

    """
    
    exact_sequence_match = 0
    correct_calls = 0

    # check if sequences are the same length
    if len(expected_tool) == len(actual_tool):
        # set a flag to see if all callings match in order
        all_calls_match = True

        for gold_call, pred_call in zip(expected_tool, actual_tool):

            # check if the function names match
            if gold_call['name'] == pred_call['name']:
                # check if the arguments match
                # arguments considered a match if in the same order
                if str(gold_call['arguments']) == str(pred_call['arguments']):
                    all_calls_match = True
            else:
                all_calls_match = False
            
        if all_calls_match:
            exact_sequence_match = 1
            call_by_call_match = 1
    
    else:
        total_calls = 0
        # Sequences differ in length -> automatically no an exact match
        # Count how many calls are correct align up to the smaller length
        min_length = min(len(expected_tool), len(actual_tool))
        for i in range(min_length):
            total_calls += 1
            if (gold_call[i]["name"] == pred_call[i]["name"]) and (gold_call[i]["arguments"] == pred_call[i]["arguments"]):
                correct_calls += 1
        call_by_call_match = correct_calls / total_calls
    

    return exact_sequence_match, call_by_call_match



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



