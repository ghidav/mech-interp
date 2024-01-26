python pca_experiment.py -tl llama-7b -hf yahma/llama-7b-hf -adp safepaca/llama-7b-hf-baseline -c resid_post -chat alpaca -bs 32
python pca_experiment.py -tl llama-7b -hf yahma/llama-7b-hf -adp safepaca/llama-7b-hf-saferpaca-Instructions-100 -c resid_post -chat alpaca -bs 32
python pca_experiment.py -tl llama-7b -hf yahma/llama-7b-hf -adp safepaca/llama-7b-hf-saferpaca-Instructions-300 -c resid_post -chat alpaca -bs 32
python pca_experiment.py -tl llama-7b -hf yahma/llama-7b-hf -adp safepaca/llama-7b-hf-saferpaca-Instructions-500 -c resid_post -chat alpaca -bs 32
python pca_experiment.py -tl llama-7b -hf yahma/llama-7b-hf -adp safepaca/llama-7b-hf-saferpaca-Instructions-1000 -c resid_post -chat alpaca -bs 32
python pca_experiment.py -tl llama-7b -hf yahma/llama-7b-hf -adp safepaca/llama-7b-hf-saferpaca-Instructions-1500 -c resid_post -chat alpaca -bs 32
python pca_experiment.py -tl llama-7b -hf yahma/llama-7b-hf -adp safepaca/llama-7b-hf-saferpaca-Instructions-2000 -c resid_post -chat alpaca -bs 32