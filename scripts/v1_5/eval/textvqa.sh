#!/bin/bash
cd /home/data/shika/LLaVA-ly-ca
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/add/llava-v1.5-7b-ca \
    --question-file /home/data/shika/LLaVA/playground/data/eval/textvqa/test.jsonl \
    --image-folder /home/data/shika/LLaVA/playground/data/eval/textvqa/train_images \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /home/data/shika/LLaVA//playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /home/data/shika/LLaVA//playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl
