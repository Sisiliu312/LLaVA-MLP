#!/bin/bash
cd /home/data/shika/LLaVA-ly-ca
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=7
MODEL=/home/data/shika/LLaVA-ly-ca/checkpoints/cat/llava-v1.5-7b


# ############# mmbench_cn #############
SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL \
    --question-file /home/data/shika/LLaVA/playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/mmbench_cn/answers/$SPLIT/llava-v1.5-13b.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /home/data/shika/LLaVA/playground/data/eval/mmbench_cn/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /home/data/shika/LLaVA/playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir /home/data/shika/LLaVA/playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir /home/data/shika/LLaVA/playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment llava-v1.5-13b

############# mmbench #############
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL \
    --question-file /home/data/shika/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /home/data/shika/LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /home/data/shika/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /home/data/shika/LLaVA/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /home/data/shika/LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-13b

############# vizwiz #############
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL \
    --question-file /home/data/shika/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /home/data/shika/LLaVA/playground/data/eval/vizwiz/test \
    --answers-file /home/data/shika/LLaVA/playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /home/data/shika/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file /home/data/shika/LLaVA/playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --result-upload-file /home/data/shika/LLaVA/playground/data/eval/vizwiz/answers_upload/llava-v1.5-13b.json

# ############# mme #############
# python -m llava.eval.model_vqa_loader \
#     --model-path $MODEL \
#     --question-file /home/data/shika/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder /home/data/shika/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
#     --answers-file /home/data/shika/LLaVA/playground/data/eval/MME/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# cd /home/data/shika/LLaVA/playground/data/eval/MME
# python convert_answer_to_mme.py --experiment llava-v1.5-13b
# cd eval_tool
# python calculation.py --results_dir answers/llava-v1.5-13b
# cd /home/data/shika/LLaVA-ly-ca

# ############# sqa #############
# python -m llava.eval.model_vqa_science \
#     --model-path $MODEL \
#     --question-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder /home/data/shika/LLaVA/playground/data/eval/scienceqa/test \
#     --answers-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_science_qa.py \
#     --base-dir /home/data/shika/LLaVA/playground/data/eval/scienceqa \
#     --result-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
#     --output-file /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
#     --output-result /home/data/shika/LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json

# ############# textvqa #############
# python -m llava.eval.model_vqa_loader \
#     --model-path $MODEL \
#     --question-file /home/data/shika/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder /home/data/shika/LLaVA/playground/data/eval/textvqa/train_images \
#     --answers-file /home/data/shika/LLaVA/playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file /home/data/shika/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file /home/data/shika/LLaVA/playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl

# ############# POPE #############
# python -m llava.eval.model_vqa_loader \
#     --model-path $MODEL \
#     --question-file /home/data/shika/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder /home/data/shika/LLaVA/playground/data/eval/pope/val2014 \
#     --answers-file /home/data/shika/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir /home/data/shika/LLaVA/playground/data/eval/pope/coco \
#     --question-file /home/data/shika/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file /home/data/shika/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl