python pca_experiment.py -tl gpt2-small -hf gpt2 -c resid_post -chat none -bs 32
python pca_experiment.py -tl gpt2-medium -hf openai-community/gpt2-medium -c resid_post -chat none -bs 32
python pca_experiment.py -tl gpt2-large -hf openai-community/gpt2-large -c resid_post -chat none -bs 32
python pca_experiment.py -tl gpt2-xl -hf openai-community/gpt2-xl -c resid_post -chat none -bs 32

python pca_experiment.py -tl Llama-2-7b-chat -hf meta-llama/Llama-2-7b-chat-hf -c resid_post -chat safe -bs 32
python pca_experiment.py -tl Llama-2-7b-chat -hf meta-llama/Llama-2-7b-chat-hf -c resid_post -chat base -bs 32
python pca_experiment.py -tl Llama-2-7b -hf meta-llama/Llama-2-7b-hf -c resid_post -chat none -bs 32

python pca_experiment.py -tl Llama-2-13b-chat -hf meta-llama/Llama-2-13b-chat-hf -c resid_post -chat safe -bs 4
python pca_experiment.py -tl Llama-2-13b-chat -hf meta-llama/Llama-2-13b-chat-hf -c resid_post -chat base -bs 4
python pca_experiment.py -tl Llama-2-13b -hf meta-llama/Llama-2-13b-hf -c resid_post -chat none -bs 4

python pca_experiment.py -tl gpt2-small -hf gpt2 -c attn_out -chat none -bs 32
python pca_experiment.py -tl gpt2-medium -hf openai-community/gpt2-medium -c attn_out -chat none -bs 32
python pca_experiment.py -tl gpt2-large -hf openai-community/gpt2-large -c attn_out -chat none -bs 32
python pca_experiment.py -tl gpt2-xl -hf openai-community/gpt2-xl -c attn_out -chat none -bs 32

python pca_experiment.py -tl gpt2-small -hf gpt2 -c mlp_out -chat none -bs 32
python pca_experiment.py -tl gpt2-medium -hf openai-community/gpt2-medium -c mlp_out -chat none -bs 32
python pca_experiment.py -tl gpt2-large -hf openai-community/gpt2-large -c mlp_out -chat none -bs 32
python pca_experiment.py -tl gpt2-xl -hf openai-community/gpt2-xl -c mlp_out -chat none -bs 32

python pca_experiment.py -tl gpt-neo-1.3B -hf EleutherAI/gpt-neo-1.3B -c resid_post -chat none -bs 8
python pca_experiment.py -tl gpt-neo-2.7B -hf EleutherAI/gpt-neo-2.7B -c resid_post -chat none -bs 8

# TODO
python pca_experiment.py -tl Llama-2-7b-chat -hf meta-llama/Llama-2-7b-chat-hf -c mlp_out -chat safe -bs 32
python pca_experiment.py -tl Llama-2-7b-chat -hf meta-llama/Llama-2-7b-chat-hf -c mlp_out -chat base -bs 32
python pca_experiment.py -tl Llama-2-7b-chat -hf meta-llama/Llama-2-7b-chat-hf -c resid_pre  -chat base -bs 32
python pca_experiment.py -tl Llama-2-7b -hf meta-llama/Llama-2-7b-hf -c mlp_out -chat none -bs 32

python pca_experiment.py -tl Llama-2-13b-chat -hf meta-llama/Llama-2-13b-chat-hf -c mlp_out -chat safe -bs 4
python pca_experiment.py -tl Llama-2-13b-chat -hf meta-llama/Llama-2-13b-chat-hf -c mlp_out -chat base -bs 4

python pca_experiment.py -tl Llama-2-7b-chat -hf meta-llama/Llama-2-7b-chat-hf -c attn_out -chat safe -bs 32
python pca_experiment.py -tl Llama-2-7b-chat -hf meta-llama/Llama-2-7b-chat-hf -c attn_out -chat base -bs 32
python pca_experiment.py -tl Llama-2-7b -hf meta-llama/Llama-2-7b-hf -c attn_out -chat none -bs 32

python pca_experiment.py -tl Llama-2-13b-chat -hf meta-llama/Llama-2-13b-chat-hf -c attn_out -chat safe -bs 4
python pca_experiment.py -tl Llama-2-13b-chat -hf meta-llama/Llama-2-13b-chat-hf -c attn_out -chat base -bs 4


# SUBSPACE ABLATION
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0 -nc 2 -c resid_post -chat none -d xs -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0.1 -nc 2 -c resid_post -chat none -d xs -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0.2 -nc 2 -c resid_post -chat none -d xs -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0 -nc 10 -c resid_post -chat none -d xs -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0.1 -nc 10 -c resid_post -chat none -d xs -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0.2 -nc 10 -c resid_post -chat none -d xs -chat none

python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0 -nc 2 -c resid_post -chat none -d full -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0.1 -nc 2 -c resid_post -chat none -d full -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0.2 -nc 2 -c resid_post -chat none -d full -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0 -nc 10 -c resid_post -chat none -d full -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0.1 -nc 10 -c resid_post -chat none -d full -chat none
python subspace_ablation.py -tl gpt2-large -hf openai-community/gpt2-large -lam 0.2 -nc 10 -c resid_post -chat none -d full -chat none

python subspace_ablation.py -tl Llama-2-7b-chat -hf meta-llama/Llama-2-7b-chat-hf -c resid_post -chat base