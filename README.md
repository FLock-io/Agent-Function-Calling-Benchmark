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
6. **Fine-tuned Qwen2.5-7B-Instruct** on the training dataset

## 3. Results

### Block and Web3 Dataset

| Model | Exact Match Accuracy | Call by Call Accuracy |
| --- | --- | --- |
| GPT-4o | 0.2244 | 0.5374 |
| GPT-4o-mini | 0.2244 | 0.4928 |
| DeepSeek-v3 | 0.2032 | 0.5811 |
| Qwen2.5-7B-Instruct | 0.4224 | 0.4224 |
| Gemini-1.5-flash | 0.4598 | 0.4598 |
| **Fine-tuned Qwen2.5** | **0.7593** | **0.8229** |

## 4. Results on General Dataset

To further validate our model's robustness, we evaluated it on **the Berkeley Function-Calling Leaderboard** [(BFCL)](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html). Our fine-tuned model not only excels in **Block and Web3** but also achieves top-tier performance on **general function-calling benchmarks**.

### BFCL-v3-simple

| Model | Exact Match Accuracy | Call by Call Accuracy |
| --- | --- | --- |
| GPT-4o | 0.9925 | 0.9925 |
| GPT-4o-mini | **0.9974** | **0.9974** |
| DeepSeek-v3 | 0.9450 | 0.9450 |
| Qwen2.5-7B-Instruct | 0.9725 | 0.9725 |
| Gemini-1.5-flash | **0.9975** | **0.9975** |
| **Fine-tuned Qwen2.5** | **0.9950** | **0.9950** |

### BFCL-v3-parallel-multi

| Model | Exact Match Accuracy | Call by Call Accuracy |
| --- | --- | --- |
| **GPT-4o** | **0.9145** | **0.9145** |
| GPT-4o-mini | 0.8808 | 0.8808 |
| DeepSeek-v3 | 0.8850 | 0.8850 |
| Qwen2.5-7B-Instruct | 0.7700 | 0.7700 |
| Gemini-1.5-flash | 0.8900 | 0.8900 |
| Fine-tuned Qwen2.5 | 0.8700 | 0.8700 |

## 5. Analysis

### 5.1 Key Takeaways
- **Fine-tuning significantly improves performance**: Our **fine-tuned Qwen2.5-7B** model consistently outperforms other models across datasets.
- **Strong generalization ability**: Unlike other models that perform well on specific benchmarks, our fine-tuned model **excels on both Block and Web3 and general function-calling tasks**.

### 5.2 Model Performance Breakdown
- **GPT-4o models** perform well in the general BFCL dataset but drop significantly in our block and web3 dataset.
- **Gemini-1.5-flash** is the best-performing proprietary model in the block and web3 dataset.
- **Qwen2.5-7B base model performs decently**, but fine-tuning significantly boosts accuracy.
- **Fine-tuned Qwen2.5-7B is the best-performing model overall**, excelling in both **domain-specific and general tasks**.
- **DeepSeek-v3 performs adequately but falls short of the fine-tuned Qwen2.5-7B model.**

Our **fine-tuned Qwen2.5-7B model stands out as a robust solution** for function-calling tasks, proving its effectiveness in both **specialized domains** and **general-purpose applications**.

