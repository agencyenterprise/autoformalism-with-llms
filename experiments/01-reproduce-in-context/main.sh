#!/bin/sh

models=(
    "gpt-4-turbo-2024-04-09"
    "gpt-4-0125-preview"
    "gpt-4-1106-preview"
    "gpt-4-0613"
    "gpt-3.5-turbo-0125"
    "gpt-3.5-turbo-1106"
)

for model in "${models[@]}"; do
    echo "Running $model"
    python run_experiment.py --name $model --model $model
    echo "Done with $model"
done
