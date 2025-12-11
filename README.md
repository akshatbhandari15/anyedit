# Investigating Long-Form Knowledge Editing Across Small Language Models

*Final Project implementation for Columbia COMS4705*

The following commands were used to run the experiments for the UnKEBench dataset. In each case, the first command runs the AnyEdit algorithm with the MEMIT backbone (as they do in the AnyEdit paper) on the specified model and hyperparameters file. We set the dataset size limit to 10 for quick evaluation.  We only perform 1 edit per run, following the AnyEdit paper. The second command computes and prints the metrics for the quality of the generation. The following metrics are computed: Bleu, Rouge-1, Rouge-2, Rouge-L, and BERTScore. 

We used an A100 GPU with 80GB for running these experiments, since the MEMIT backbone requires significant GPU memory.

## Gemma3-1B-it

`python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=google/gemma-3-1b-it   --hparams_fname=Gemma3-1B-it.json   --ds_name=unke   --dataset_size_limit=10   --num_edits=1`

`python -m experiments.summarize_uns --file_path=/home/petros/msc/nlp/project/AnyEdit/output/MEMIT_ARE_google_gemma-3-1b-it_unke_result.json`

`python3 -m experiments.evaluate_edited_model   --alg_name=MEMIT_ARE   --model_name=google/gemma-3-1b-it  --hparams_fname=Gemma3-1B-it.json   --edit_ds_name=unke   --edit_size_limit=10   --gsm8k_size=200   --skip_mmlu   --output_dir=output/gemma-1b   --save_model=models/Gemma3-1B-it.json/`


## Gemma3-4B-it

`python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=google/gemma-3-4b-it   --hparams_fname=Gemma3-4B-it.json   --ds_name=unke   --dataset_size_limit=10   --num_edits=1`

`python -m experiments.summarize_uns --file_path=/home/petros/msc/nlp/project/AnyEdit/output/MEMIT_ARE_google_gemma-3-4b-it_unke_result.json`

`python3 -m experiments.evaluate_edited_model   --alg_name=MEMIT_ARE   --model_name=google/gemma-3-4b-it  --hparams_fname=Gemma3-1B-it.json   --edit_ds_name=unke   --edit_size_limit=10   --gsm8k_size=200   --skip_mmlu   --output_dir=output/gemma-4b`

## Qwen2.5-3B-Instruct

`python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=Qwen/Qwen2.5-3B-Instruct   --hparams_fname=Qwen2.5-3B-Instruct.json   --ds_name=unke   --dataset_size_limit=10   --num_edits=1`

`python -m experiments.summarize_uns --file_path=/home/petros/msc/nlp/project/AnyEdit/output/MEMIT_ARE_Qwen2.5-3B-Instruct_unke_result.json`

## Lamma 3.2-3B

`python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=meta-llama/Llama-3.2-3B   --hparams_fname=Llama3.2-3B.json   --ds_name=unke   --dataset_size_limit=10   --num_edits=1`

`python -m experiments.summarize_uns --file_path=/home/petros/msc/nlp/project/AnyEdit/output/MEMIT_ARE_meta-llama_Llama-3.2-3B_unke_result.json`


# Qwen 2.5-7B
`python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=Qwen/Qwen2.5-7B-Instruct --hparams_fname=Qwen2.5-7B-Instruct.json   --ds_name=unke   --dataset_size_limit=10   --num_edits=1 --no_edit`

`python3 -m experiments.evaluate_uns  --alg_name=MEMIT_ARE   --model_name=Qwen/Qwen2.5-7B-Instruct   --hparams_fname=Qwen2.5-7B-Instruct.json   --dataset_size_limit=200   --output_dir=output/qwen2_5/`


# Qwen 2.5-3B

`python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=Qwen/Qwen2.5-7B-Instruct --hparams_fname=Qwen2.5-7B-Instruct.json   --ds_name=unke   --dataset_size_limit=10   --num_edits=1`

# Meta-Llama-3-8B-Instruct

`python3 -m experiments.evaluate_uns     --alg_name=MEMIT_ARE     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B-Instruct.json     --ds_name=unke     --dataset_size_limit=10     --num_edits=1`

`python3 -m experiments.evaluate_edited_model   --alg_name=MEMIT_ARE   --model_name=meta-llama/Meta-Llama-3-8B-Instruct   --hparams_fname=Llama3-8B-Instruct.json   --edit_ds_name=unke   --edit_size_limit=10   --gsm8k_size=200   --skip_mmlu   --output_dir=output/   --save_model=models/Llama-3-8b/`

`python3 -m experiments.evaluate_gsm8k   --alg_name=MEMIT_ARE   --model_name=meta-llama/Meta-Llama-3-8B-Instruct   --hparams_fname=Llama3-8B-Instruct.json   --dataset_size_limit=200   --output_dir=output/`