# Agent-Function-Calling-Benchmark

## Usage

### 1. Testing Proprietary Models (e.g., OpenAI)
- Create a file named `.env` in the project directory.
- Add your OpenAI API key to the `.env` file:
  ```env
  OPENAI_API_KEY=<your_api_key>
  ```
- Prepare the input data in the format of [example_data.jsonl](./data/example_data.jsonl), including the input query, function call list, and expected answers.
- Run the script:
  ```sh
  python run_baseline.py
  ```

### 2. Testing Custom Models
- Prepare your models' predictions along with the gold (reference) answers in the format of [results.jsonl](./data/baseline_gpt-4o-mini_results.jsonl).
- Run the script:
  ```sh
  python run_evaluation.py
  ```

# Experiment Report: Function Call Benchmark on Block and Web3 Dataset

## 1. Introduction

This report presents the evaluation of various language models on a function call dataset related to **Block and Web3**. The dataset comprises **187 test samples**, and the evaluation focuses on two key benchmarks:

- **Exact Match Accuracy (exact_match_acc)**: The generated ordered function call, including arguments, is entirely correct.
- **Call by Call Accuracy (call_by_call_acc)**: The generated ordered function call contains some correct arguments, even if the entire sequence is not perfect. For example, if two function calls are required but the model only correctly generates the first one, it still receives partial credit.

### Dataset Description
Each data instance includes a **query** and a list of **available tools**. The model must generate function calls using the provided tools to correctly respond to the query.

#### Example Data Format:
```json
{
  "query": "Track crosschain message verification, implement timeout recovery procedures.",
  "answers": [
    {"name": "track_crosschain_message", "arguments": {"message_id": "msg12345"}},
    {"name": "schedule_timeout_check", "arguments": {"message_id": "msg12345", "timeout": "30"}}
  ],
  "tools": [
    {"type": "function", "function": {"name": "track_crosschain_message", "description": "Track the status of a crosschain message", "parameters": {"type": "object", "properties": {"message_id": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "schedule_timeout_check", "description": "Schedule a timeout check for a message", "parameters": {"type": "object", "properties": {"message_id": {"type": "string"}, "timeout": {"type": "integer"}}}}}
  ]
}
```

## 2. Methodology

We evaluated multiple models using different inference methods:

1. **GPT-4o**
2. **GPT-4o-mini**
3. **Qwen2.5-7B-Instruct**
4. **DeepSeek-v3**
5. **Gemini-1.5-flash**
6. **Gemini-2.9-flash**
7. **Fine-tuned Qwen2.5-7B-Instruct** on the training dataset

## 3. Results

### Block and Web3 Dataset

| Model | Exact Match Accuracy | Call by Call Accuracy |
| --- | --- | --- |
| **Proprietary Models**
| GPT-4o | 0.4598 | 0.6624 |
| GPT-4o-mini | 0.3529 | 0.5179 |
| Gemini-1.5-flash | 0.4438 | 0.5351 |
| Gemini-2.0-flash | 0.3957 | 0.4924 |
| **Open-Sourced Models** 
| DeepSeek-v3 | 0.2887 | 0.6229 |
| Qwen2.5-7B-Instruct | 0.5250 | 0.5790 |
| **Fine-tuned Qwen2.5** | **0.7593** | **0.8229** |

## 4. Results on General Dataset

To further validate our model's robustness, we evaluated it on **the Berkeley Function-Calling Leaderboard** [(BFCL)](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html). Our fine-tuned model not only excels in **Block and Web3** but also achieves top-tier performance on **general function-calling benchmarks**.

### BFCL-v3-simple

| Model | Exact Match Accuracy | Call by Call Accuracy |
| --- | --- | --- |
| **Proprietary Models**
| GPT-4o | 0.9925 | 0.9925 |
| GPT-4o-mini | **0.9974** | **0.9974** |
| Gemini-1.5-flash | **0.9975** | **0.9975** |
| Gemini-2.0-flash | 0.9938 | 0.9938 |
| **Open-Sourced Models** 
| DeepSeek-v3 | 0.9450 | 0.9450 |
| Qwen2.5-7B-Instruct | 0.9725 | 0.9725 |
| **Fine-tuned Qwen2.5** | **0.9950** | **0.9950** |

**Noted** that the BFCL-v3-simple dataset includes queries that require one-step function calling so the Exact Match Accuracy and Cally by Call Accuracy are the same.

### BFCL-v3-parallel-multi

| Model | Exact Match Accuracy | Call by Call Accuracy |
| --- | --- | --- |
| **Proprietary Models**
| **GPT-4o** | **0.9393** | **0.9444** |
| GPT-4o-mini | 0.9343 | 0.9343 |
| Gemini-1.5-flash | 0.9251 | 0.9358 |
| Gemini-2.0-flash | wait | wait |
| **Open-Sourced Models** 
| DeepSeek-v3 | wait | wait |
| Qwen2.5-7B-Instruct | 0.7700 | 0.7700 |
| Fine-tuned Qwen2.5 | 0.8900 | 0.8925 |

## 5. Analysis and Discussion

### Performance Trends
- Proprietary models such as **GPT-4o** and **Gemini-1.5-flash** demonstrate strong generalization across datasets, achieving high accuracy on both Block/Web3 tasks and BFCL benchmarks.
- **Fine-tuned Qwen2.5** outperforms all other open-source models, especially in the Block/Web3 domain, suggesting that domain-specific tuning significantly enhances function-calling capabilities.

### Strengths of Fine-tuned Models
- Fine-tuning Qwen2.5 on a domain-specific dataset resulted in a **45% improvement in exact match accuracy** compared to its pre-trained version.
- The improvement in **call-by-call accuracy** suggests that fine-tuning helps the model understand structured outputs more precisely, even when full correctness is not achieved.

### Limitations and Challenges
- Open-source models like **DeepSeek-v3** struggle with **exact function sequence generation**, indicating a need for better structured data training.
- Proprietary models still outperform fine-tuned open-source models on **general function-calling tasks**, suggesting that larger-scale pretraining and reinforcement tuning contribute significantly to performance.

## 6. Conclusion
This benchmark highlights the effectiveness of fine-tuning in improving function-calling accuracy for **Block and Web3** tasks. While proprietary models retain an edge in generalization, domain-specific fine-tuning presents a viable pathway for enhancing open-source models' capabilities.

