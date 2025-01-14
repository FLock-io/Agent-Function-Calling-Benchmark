# Agent-Function-Calling-Benchmark

## Usage

### 1. Testing Proprietary Models (e.g., OpenAI)
- Create a file named `.env` in the project directory.
- Add your OpenAI API key to the `.env` file:
  ```env
  OPENAI_API_KEY=<your_api_key>
- Prepare the input data in the format of [example_data.jsonl](./data/example_data.jsonl), including the input query, function call list as well as the expected answers.
- Run the script
  ```
  python run_baseline.py
### 2. Testing Custom Models
- Prepare your models' predictions along with the gold (reference) answers in the format of [results.jsonl](./data/baseline_gpt-4o-mini_results.jsonl)
- Run the script
```
  python run_evaluation.py