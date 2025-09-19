seed=$1

uv run fp_iteration.py --model_name allenai/OLMo-2-0425-1B-Instruct --seed $seed -n olmo2-1b-inst_seed$seed
uv run fp_iteration.py --model_name allenai/OLMo-2-0425-1B --seed $seed -n olmo2-1b-base_seed$seed
uv run fp_iteration.py --model_name allenai/OLMo-2-1124-7B-Instruct --seed $seed -n olmo2-7b-inst_seed$seed
uv run fp_iteration.py --model_name allenai/OLMo-2-1124-7B --seed $seed -n olmo2-7b-base_seed$seed
uv run fp_iteration.py --model_name allenai/OLMo-2-1124-13B-Instruct --seed $seed -n olmo2-13b-inst_seed$seed
uv run fp_iteration.py --model_name allenai/OLMo-2-1124-13B --seed $seed -n olmo2-13b-base_seed$seed

uv run fp_iteration.py --model_name Qwen/Qwen3-0.6B --seed $seed -n qwen3-0.6b-inst_seed$seed
uv run fp_iteration.py --model_name Qwen/Qwen3-0.6B-Base --seed $seed -n qwen3-0.6b-base_seed$seed
uv run fp_iteration.py --model_name Qwen/Qwen3-1.7B --seed $seed -n qwen3-1.7b-inst_seed$seed
uv run fp_iteration.py --model_name Qwen/Qwen3-1.7B-Base --seed $seed -n qwen3-1.7b-base_seed$seed
uv run fp_iteration.py --model_name Qwen/Qwen3-4B --seed $seed -n qwen3-4b-inst_seed$seed
uv run fp_iteration.py --model_name Qwen/Qwen3-4B-Base --seed $seed -n qwen3-4b-base_seed$seed
uv run fp_iteration.py --model_name Qwen/Qwen3-8B --seed $seed -n qwen3-8b-inst_seed$seed
uv run fp_iteration.py --model_name Qwen/Qwen3-8B-Base --seed $seed -n qwen3-8b-base_seed$seed
uv run fp_iteration.py --model_name Qwen/Qwen3-14B --seed $seed -n qwen3-14b-inst_seed$seed
uv run fp_iteration.py --model_name Qwen/Qwen3-14B-Base --seed $seed -n qwen3-14b-base_seed$seed

uv run fp_iteration.py --model_name google/gemma-3-270m-it --seed $seed -n gemma3-270m-inst_seed$seed
uv run fp_iteration.py --model_name google/gemma-3-270m --seed $seed -n gemma3-270m-base_seed$seed
uv run fp_iteration.py --model_name google/gemma-3-1b-it --seed $seed -n gemma3-1b-inst_seed$seed
uv run fp_iteration.py --model_name google/gemma-3-1b-pt --seed $seed -n gemma3-1b-base_seed$seed
