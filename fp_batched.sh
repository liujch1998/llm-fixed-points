seed=$1

uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name allenai/OLMo-2-0425-1B-Instruct --batch_size 1 --seed $seed -n olmo2-1b-inst_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name allenai/OLMo-2-0425-1B --batch_size 1 --seed $seed -n olmo2-1b-base_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name allenai/OLMo-2-1124-7B-Instruct --batch_size 1 --seed $seed -n olmo2-7b-inst_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name allenai/OLMo-2-1124-7B --batch_size 1 --seed $seed -n olmo2-7b-base_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name allenai/OLMo-2-1124-13B-Instruct --batch_size 1 --seed $seed -n olmo2-13b-inst_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name allenai/OLMo-2-1124-13B --batch_size 1 --seed $seed -n olmo2-13b-base_seed$seed

uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-0.6B --batch_size 1 --seed $seed -n qwen3-0.6b_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-0.6B-Base --batch_size 1 --seed $seed -n qwen3-0.6b-base_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-1.7B --batch_size 1 --seed $seed -n qwen3-1.7b_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-1.7B-Base --batch_size 1 --seed $seed -n qwen3-1.7b-base_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-4B --batch_size 1 --seed $seed -n qwen3-4b_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-4B-Base --batch_size 1 --seed $seed -n qwen3-4b-base_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-8B --batch_size 1 --seed $seed -n qwen3-8b_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-8B-Base --batch_size 1 --seed $seed -n qwen3-8b-base_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-14B --batch_size 1 --seed $seed -n qwen3-14b_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name Qwen/Qwen3-14B-Base --batch_size 1 --seed $seed -n qwen3-14b-base_seed$seed

uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name google/gemma-3-270m-it --batch_size 1 --seed $seed -n gemma3-270m-it_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name google/gemma-3-270m --batch_size 1 --seed $seed -n gemma3-270m_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name google/gemma-3-1b-it --batch_size 1 --seed $seed -n gemma3-1b-it_seed$seed
uv run fp_batched.py --dtype fp32 --scheduler_patience 1000 --detach --eval_every 1000 --eval_len 100 --model_name google/gemma-3-1b-pt --batch_size 1 --seed $seed -n gemma3-1b-pt_seed$seed
