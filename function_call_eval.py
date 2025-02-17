import json
import re
import string


# Function to normalize the answer, perhaps we can use this to normalize the function names if needed
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_valid_function(func):
    """
    Check if a function is valid by checking if it has a name and arguments
    """
    if isinstance(func, dict):
        if "name" in func and "arguments" in func:
            return True
    return False


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
    call_by_call_match = 0

    # check if both sequences are valid
    expected_tool = [func for func in expected_tool if is_valid_function(func)]
    actual_tool = [func for func in actual_tool if is_valid_function(func)]

    # check if sequences are the same length
    if len(expected_tool) == len(actual_tool):
        # set a flag to see if all callings match in order
        all_calls_match = True

        for gold_call, pred_call in zip(expected_tool, actual_tool):

            # check if the function names match
            if gold_call["name"] == pred_call["name"]:
                # check if the arguments match
                # arguments considered a match if in the same order
                gold_arguments = gold_call["arguments"]
                pred_arguments = pred_call["arguments"]
                if json.dumps(gold_arguments, ensure_ascii=False) == json.dumps(
                    pred_arguments, ensure_ascii=False
                ):
                    all_calls_match = True
            else:
                all_calls_match = False

        if all_calls_match:
            exact_sequence_match = 1
            call_by_call_match = 1

    else:
        total_calls = 0
        correct_calls = 0
        # Sequences differ in length -> automatically no an exact match
        # Count how many calls are correct align up to the smaller length
        min_length = min(len(expected_tool), len(actual_tool))
        for i in range(min_length):
            total_calls += 1
            if (expected_tool[i]["name"] == actual_tool[i]["name"]) and (
                json.dumps(expected_tool[i]["arguments"], ensure_ascii=False) == json.dumps(actual_tool[i]["arguments"], ensure_ascii=False)
            ):
                correct_calls += 1

        call_by_call_match = correct_calls / total_calls if total_calls > 0 else 0

    return exact_sequence_match, call_by_call_match
