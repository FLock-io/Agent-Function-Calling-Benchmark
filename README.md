# Agent-Function-Calling-Benchmark

## Usage

### 1. Testing Proprietary Models (e.g., OpenAI)

- Create a file named `.env` in the project directory.
- Add your OpenAI API key to the `.env` file:
  ```env
  OPENAI_API_KEY=<your_api_key>
  ```
- Prepare the input data in the format of [example\_data.jsonl](./data/example_data.jsonl), including the input query, function call list, and expected answers.
- Run the script:
  ```sh
  python run_baseline.py
  ```

### 2. Testing Custom Models

- Prepare your models' predictions along with the reference answers in the format of [results.jsonl](./data/baseline_gpt-4o-mini_results.jsonl).
- Run the script:
  ```sh
  python run_evaluation.py
  ```

# Experiment Report: Function Call Benchmark on Block and Web3 Dataset

## 1. Introduction

This report presents an evaluation of various language models on a function call dataset related to **Block and Web3**. The dataset comprises **187 test samples**, and the evaluation focuses on **Exact Match Accuracy**:

- **Exact Match Accuracy (exact\_match\_acc)**: Measures correctness of the generated function calls, including order and arguments.

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
6. **Gemini-2.0-flash**
7. **Our fine-tuned Qwen2.5-7B-Instruct** on the training dataset

## 3. Results

### Block and Web3 Dataset

| Model                   | Exact Match Accuracy |
| ----------------------- | -------------------- |
| **Proprietary Models**  |                      |
| GPT-4o                  | 0.4598               |
| GPT-4o-mini             | 0.3529               |
| Gemini-1.5-flash        | 0.4438               |
| Gemini-2.0-flash        | 0.3957               |
| **Open-Sourced Models** |                      |
| DeepSeek-v3             | 0.2887               |
| Qwen2.5-7B-Instruct     | 0.3100               |
| \*\*Ours\*\*            | **0.7593**           |

## 4. Results on General Dataset

To validate model robustness, we evaluated it on **the Berkeley Function-Calling Leaderboard (BFCL)** [(BFCL)](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html). The fine-tuned Qwen2.5 model demonstrated competitive performance.

### BFCL-v3-simple

| Model                   | Exact Match Accuracy |
| ----------------------- | -------------------- |
| **Proprietary Models**  |                      |
| GPT-4o                  | 0.9925               |
| GPT-4o-mini             | **0.9974**           |
| Gemini-1.5-flash        | **0.9975**           |
| Gemini-2.0-flash        | 0.9938               |
| **Open-Sourced Models** |                      |
| DeepSeek-v3             | 0.9450               |
| Qwen2.5-7B-Instruct     | 0.9725               |
| Ours                    | **0.9950**           |

### BFCL-v3-parallel-multi

| Model                   | Exact Match Accuracy |
| ----------------------- | -------------------- |
| **Proprietary Models**  |                      |
| GPT-4o                  | **0.9393**           |
| GPT-4o-mini             | 0.9343               |
| Gemini-1.5-flash        | 0.9251               |
| Gemini-2.0-flash        | 0.9161               |
| **Open-Sourced Models** |                      |
| DeepSeek-v3             | 0.9300               |
| Qwen2.5-7B-Instruct     | 0.7700               |
| Ours                    | 0.8900               |

## 5. Analysis and Discussion

### Performance Trends

- Our fine-tuned Qwen2.5 significantly outperforms other open-source models in the **Block/Web3** domain, achieving an exact match accuracy of **0.7593**, a remarkable improvement over the pre-trained version (0.3100).
- Our model maintains high performance in general function-calling tasks, achieving **0.9950** on the **BFCL-v3-simple** benchmark and **0.8900** on **BFCL-v3-parallel-multi**, making it a competitive alternative to proprietary models.
- Proprietary models (**GPT-4o, Gemini-1.5-flash**) continue to dominate general benchmarks, but our fine-tuned model closes the gap while excelling in domain-specific tasks.

### Strengths of Fine-tuned Models

- Fine-tuning Qwen2.5 on a domain-specific dataset led to a **145% improvement** in exact match accuracy in the **Block/Web3** domain.
- The model sustains high accuracy in general function-calling tasks, making it both **versatile and specialized**.

### Limitations and Challenges

- Open-source models like **DeepSeek-v3** struggle with **exact function sequence generation**, highlighting the need for better structured data training.
- Proprietary models still lead in **general function-calling tasks**, suggesting that large-scale pretraining and reinforcement tuning play a crucial role.

## 6. Conclusion

This benchmark demonstrates that **fine-tuning** is highly effective in improving function-calling accuracy for **Block and Web3** tasks. Our model not only **excels in domain-specific applications** but also maintains **strong performance in general function-calling tasks**, presenting a viable alternative to proprietary solutions.

