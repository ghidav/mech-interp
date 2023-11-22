[1mdiff --git a/utils.py b/utils.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mindex 3360d1e..c1198c2[m
[1m--- a/utils.py[m
[1m+++ b/utils.py[m
[36m@@ -39,8 +39,9 @@[m [mdef list_to_str(x):[m
 [m
     return s[:-1][m
 [m
[31m-def load_model(hf_model_name, base_model="", adapter_model="", device='cpu', n_devices=1, dtype=torch.float32):[m
[32m+[m[32mdef load_model(hf_model_name, tl_model_name="", adapter_model="", device='cpu', n_devices=1, dtype=torch.float32):[m
     model = None[m
[32m+[m[32m    if tl_model_name =="": tl_model_name = hf_model_name[m
 [m
     if adapter_model != "":[m
         hf_model = AutoModelForCausalLM.from_pretrained([m
[36m@@ -52,11 +53,12 @@[m [mdef load_model(hf_model_name, base_model="", adapter_model="", device='cpu', n_d[m
         del hf_model[m
 [m
         tokenizer = AutoTokenizer.from_pretrained(hf_model_name)[m
[31m-        model = HookedTransformer.from_pretrained(base_model, hf_model=peft_model, tokenizer=tokenizer,[m
[32m+[m[32m        model = HookedTransformer.from_pretrained(tl_model_name, hf_model=peft_model, tokenizer=tokenizer,[m
                                                     device=device, n_devices=n_devices, dtype=dtype)[m
     else:[m
[31m-        try: [m
[31m-            model = HookedTransformer.from_pretrained(hf_model, device=device, n_devices=n_devices, dtype=dtype)[m
[32m+[m[32m        try:[m
[32m+[m[32m            model = HookedTransformer.from_pretrained(tl_model_name, device=device, n_devices=n_devices, dtype=dtype)[m
[32m+[m[32m            print("Loaded model into HookedTransformer")[m
         except Exception as e:[m
             print(e)[m
 [m
[36m@@ -64,8 +66,10 @@[m [mdef load_model(hf_model_name, base_model="", adapter_model="", device='cpu', n_d[m
             try:[m
                 tokenizer = AutoTokenizer.from_pretrained(hf_model_name)[m
                 hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True)[m
[31m-                model = HookedTransformer.from_pretrained(base_model, hf_model=hf_model, tokenizer=tokenizer,[m
[32m+[m[32m                print("Loaded model from hf. Attempting to load it to HookedTransformer")[m
[32m+[m[32m                model = HookedTransformer.from_pretrained(tl_model_name, hf_model=hf_model, tokenizer=tokenizer,[m
                                                         device=device, n_devices=n_devices, dtype=dtype)[m
[32m+[m[32m                print("Loaded model into HookedTransformer")[m
                 del hf_model[m
 [m
             except Exception as e:[m
