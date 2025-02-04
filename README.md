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

## 2. Methodology

We evaluated multiple models using different inference methods:

1. **GPT-4o** (Chat mode and Function Call mode)
2. **GPT-4o-mini** (Chat mode and Function Call mode)
3. **Qwen2.5-7B-Instruct** (Base model)
4. **Fine-tuned Qwen2.5-7B-Instruct** on training dataset (with different learning rates)
5. **Fine-tuned DeepSeek-R1-7B**
6. **DeepSeek-Chat** (Function Call mode)
7. **O1 Model** (Function Call mode)

The function call tests were performed using both **chat-based interactions** and **structured function-call outputs**, depending on model capabilities.

## 3. Results

The evaluation results for the **Block and Web3** dataset are presented in the table below:

| Model | Exact Match Accuracy | Call by Call Accuracy |
| --- | --- | --- |
| GPT-4o (Chat) | 0.3315 | 0.3315 |
| GPT-4o-mini (Chat) | 0.2768 | 0.2768 |
| GPT-4o (Function Call) | 0.2244 | 0.5374 |
| GPT-4o-mini (Function Call) | 0.2244 | 0.4928 |
| Qwen2.5-7B-Instruct | 0.4224 | 0.4224 |
| Fine-tuned Qwen2.5 (lr=1e-5) | 0.5935 | 0.6922 |
| Fine-tuned Qwen2.5 (lr=1e-4) | 0.7593 | 0.8229 |
| Fine-tuned DeepSeek-R1-7B | 0.6577 | 0.7549 |
| O1 (Function Call) | 0.2673 | 0.2673 |
| DeepSeek-Chat (Function Call) | 0.2032 | 0.5811 |

## 4. Analysis

### 4.1 General Observations
- **Fine-tuning significantly improves performance**: The fine-tuned **Qwen2.5-7B** models outperform all other models, especially at **lr=1e-4**, where they achieve **0.7593 exact match accuracy and 0.8229 call by call accuracy**.
- **Function call accuracy improves with structured function-call inference**: While chat-based inference provides decent results, function-call-based inference generally achieves **higher call-by-call accuracy** (e.g., GPT-4o function call at **0.5374** vs. GPT-4o chat at **0.3315**).
- **DeepSeek models underperform in exact match but show potential in call-by-call accuracy**: **DeepSeek-Chat** (function-call) only scores **0.2032 exact match accuracy**, but performs significantly better in call-by-call accuracy (**0.5811**), indicating partial correctness in outputs.

### 4.2 Model Performance Breakdown
- **GPT-4o models** perform well in chat mode but drop significantly in exact match accuracy when switching to function-call mode.
- **Qwen2.5-7B base model performs decently**, but fine-tuning significantly boosts accuracy.
- **Fine-tuned Qwen2.5-7B (lr=1e-4) is the best-performing model overall**, achieving the highest accuracy scores.
- **DeepSeek-R1-7B fine-tuned performs well, but not as strong as Qwen2.5-7B fine-tuned models.**
- **O1 and DeepSeek-Chat models perform the worst**, showing lower scores in both metrics.

