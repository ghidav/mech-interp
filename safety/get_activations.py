from transformer_lens import HookedTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch as th
from tqdm.auto import tqdm
import pandas as pd

# Model loading

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
dtype = th.bfloat16

hfm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
safe_model = HookedTransformer.from_pretrained_no_processing("Llama-2-7b-chat", hf_model=hfm, n_devices=4, tokenizer=tokenizer, dtype=dtype)
del hfm

hfm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
unsafe_model = HookedTransformer.from_pretrained_no_processing("Llama-2-7b", hf_model=hfm, n_devices=4, tokenizer=tokenizer, dtype=dtype)
del hfm

safe_model.tokenizer.padding_side = 'left'

safe_model.eval()
unsafe_model.eval()

safe_model.tokenizer.pad_token = safe_model.tokenizer.eos_token
safe_model.tokenizer.pad_token_id = safe_model.tokenizer.eos_token_id

nl = len(safe_model.blocks)

def convert_to_chat(model, prompts, sys_prompt=False):
    new_prompts = []
    if sys_prompt:
        content = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    else:
        content = "You are a helpful assistant"
        
    for p in prompts:
        tokens = model.tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": content,
            },
            {"role": "user", "content": p}
        ])

        new_prompts.append(model.to_string(tokens))

    return new_prompts

def get_activations(model, prompts, component, bs=32):
    nl = len(model.blocks)
    activations = []

    for b in tqdm(range(len(prompts) // bs)):
        tokens = model.to_tokens(prompts[b*bs:(b+1)*bs])

        with th.no_grad():
            _, cache = model.run_with_cache(tokens)

        activations.append(th.cat([cache[f'blocks.{l}.hook_{component}'][:, None, -1, :].cpu() for l in range(nl)], 1)) # [b l dm]
        del cache

    return th.cat(activations, 0) # [np l dm]


blocked_prompts = pd.read_csv(f"data/full_prompts_blocked.csv", index_col=0).reset_index(drop=True)
N = len(blocked_prompts)
bs = 32

blocked_prompts = pd.concat([
    blocked_prompts.iloc[:(N - N%bs) // 2],
    blocked_prompts.iloc[N//2:N - N%bs // 2]
])

nb = len(blocked_prompts) // bs

chat_blocked_prompts_nosys = convert_to_chat(safe_model, blocked_prompts['prompt'], sys_prompt=False)
chat_blocked_prompts_sys = convert_to_chat(safe_model, blocked_prompts['prompt'], sys_prompt=True)

component = 'resid_pre'

chat_blocked_nosys_activations = get_activations(safe_model, chat_blocked_prompts_nosys, component).cpu()
th.save(chat_blocked_nosys_activations, f"activations/blocked_chat_nosys_{component}.pt")

chat_blocked_sys_activations = get_activations(safe_model, chat_blocked_prompts_sys, component, bs=8).cpu()
th.save(chat_blocked_sys_activations, f"activations/blocked_chat_sys_{component}.pt")

chat_blocked_raw_activations = get_activations(safe_model, blocked_prompts['prompt'].tolist(), component).cpu()
th.save(chat_blocked_raw_activations, f"activations/blocked_chat_raw_{component}.pt")

unsafe_blocked_activations = get_activations(unsafe_model, blocked_prompts['prompt'].tolist(), component).cpu()
th.save(unsafe_blocked_activations, f"activations/blocked_unsafe_{component}.pt")