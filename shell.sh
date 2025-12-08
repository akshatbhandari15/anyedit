sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 python3.10-venv python3.10-dev
python3.10 -m venv .venv
source .venv/bin/activate
git clone https://github.com/pgiouroukis/anyedit.git
cd anyedit
pip install -r requirements.txt
hf auth login
cd AnyEdit
python3 -m experiments.evaluate_uns   --alg_name=MEMIT_ARE   --model_name=google/gemma-3-1b-it   --hparams_fname=Gemma3-1B-it.json   --ds_name=unke   --dataset_size_limit=1000   --num_edits=1

python -m experiments.summarize_uns --file_path=/home/petros/msc/nlp/project/AnyEdit/output/MEMIT_ARE_Gemma3-1B-IT_unke_result.json