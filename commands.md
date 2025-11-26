
## Qwen2.5-3B-Instruct

python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=Qwen/Qwen2.5-3B-Instruct   --hparams_fname=Qwen2.5-3B-Instruct.json   --ds_name=unke   --dataset_size_limit=1   --num_edits=1

python -m experiments.summarize_uns --file_path=/home/petros/msc/nlp/project/AnyEdit/output/MEMIT_ARE_Qwen2.5-3B-Instruct_unke_result.json

## Gemma3-1B-it

python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=google/gemma-3-1b-it   --hparams_fname=Gemma3-1B-it.json   --ds_name=unke   --dataset_size_limit=1   --num_edits=1

python -m experiments.summarize_uns --file_path=/home/petros/msc/nlp/project/AnyEdit/output/MEMIT_ARE_google_gemma-3-1b-it_unke_result.json

## Gemma3-4B-it

python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=google/gemma-3-4b-it   --hparams_fname=Gemma3-4B-it.json   --ds_name=unke   --dataset_size_limit=1   --num_edits=1

python -m experiments.summarize_uns --file_path=/home/petros/msc/nlp/project/AnyEdit/output/MEMIT_ARE_google_gemma-3-4b-it_unke_result.json

## Lamma 3.2-3B

python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=meta-llama/Llama-3.2-3B   --hparams_fname=Llama3.2-3B.json   --ds_name=unke   --dataset_size_limit=1   --num_edits=1

python -m experiments.summarize_uns --file_path=/home/petros/msc/nlp/project/AnyEdit/output/MEMIT_ARE_meta-llama_Llama-3.2-3B_unke_result.json