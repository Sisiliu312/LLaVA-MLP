#!/bin/bash
cd /home/data/shika/LLaVA-ly-ca
export PYTHONWARNINGS="ignore"


python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/cat/llava-v1.5-7b-ca \
    --question-file /home/data/shika/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/data/shika/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /home/data/shika/LLaVA/playground/data/eval/pope/coco \
    --question-file /home/data/shika/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /home/data/shika/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
