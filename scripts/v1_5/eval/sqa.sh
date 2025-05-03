#!/bin/bash
cd /home/data/shika/LLaVA-ly-ca
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/cat/llava-v1.5-7b-ca \
    --question-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /home/data/shika/LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /home/data/shika/LLaVA/playground/data/eval/scienceqa \
    --result-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json
