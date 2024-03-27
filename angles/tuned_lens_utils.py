from tuned_lens.causal import extract_causal_bases
from tuned_lens.nn.lenses import TunedLens, LogitLens, Unembed
from transformer_lens import HookedTransformer 
import torch
from typing import cast, Optional
import math
import torch as th
import torch.distributions as D
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
from functools import partial

def extract_cb(model, prompts, lens, k=10):
    cb_energies = []
    cb_vectors = []

    k = 10

    for i in range(len(prompts)):
        tokens = model.to_tokens(prompts[i])
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        resid_post = torch.cat([cache[f'blocks.{l}.hook_resid_post'] for l in range(n_layers)]) # [b p dm]

        vec = []
        ene = []
        
        for j in extract_causal_bases(lens, resid_post, k=k):
            vec.append(j.vectors[None, None, ...])
            ene.append(j.energies[None, None, ...])

        cb_vectors.append(torch.cat(vec, dim=1))
        cb_energies.append(torch.cat(ene, dim=1))

    cb_vectors = torch.cat(cb_vectors, dim=0) # [prompt layer d_model k]
    cb_energies = torch.cat(cb_energies, dim=0) # [prompt layer d_model k]

    return cb_vectors, cb_energies

"""
What we used before 

def weighted_geom_mean(p, w):
    return torch.exp((w * torch.log(p)).sum(-1) / w.sum(-1))

def aitchison_weighted_similarity(p, q, w):
    return (w * torch.log(p / weighted_geom_mean(p, w).unsqueeze(1)) * torch.log(q / weighted_geom_mean(q, w).unsqueeze(1))).sum(-1)
"""

def aitchison(
    log_p: th.Tensor,
    log_q: th.Tensor,
    *,
    weight: Optional[th.Tensor] = None,
    dim: int = -1
) -> th.Tensor:
    """Compute the (weighted) Aitchison inner product between log probability vectors.
    The `weight` parameter can be used to downweight rare tokens in an LM's vocabulary.
    See 'Changing the Reference Measure in the Simplex and Its Weighting Effects' by
    Egozcue and Pawlowsky-Glahn (2016) for discussion.
    """
    # Normalize the weights to sum to 1 if necessary
    if weight is not None:
        weight = weight / weight.sum(dim=dim, keepdim=True)

    # Project to Euclidean space...
    x = _clr(log_p, weight, dim=dim)
    y = _clr(log_q, weight, dim=dim)

    # Then compute the weighted dot product
    return _weighted_mean(x * y, weight, dim=dim)


def aitchison_similarity(
    log_p: th.Tensor,
    log_q: th.Tensor,
    *,
    weight: Optional[th.Tensor] = None,
    dim: int = -1,
    eps: float = 1e-8
) -> th.Tensor:
    """Cosine similarity of log probability vectors with the Aitchison inner product.
    Specifically, we compute <p, q> / max(||p|| * ||q||, eps), where ||p|| is the norm
    induced by the Aitchison inner product: sqrt(<p, p>).
    """
    affinity = aitchison(log_p, log_q, weight=weight, dim=dim)
    norm_p = aitchison(log_p, log_p, weight=weight, dim=dim).sqrt()
    norm_q = aitchison(log_q, log_q, weight=weight, dim=dim).sqrt()
    return affinity / (norm_p * norm_q).clamp_min(eps)


def _clr(
    log_y: th.Tensor, weight: Optional[th.Tensor] = None, dim: int = -1
) -> th.Tensor:
    """Apply a (weighted) centered logratio transform to a log probability vector.
    This is equivalent to subtracting the geometric mean in log space, and it is one of
    three main isomorphisms between the simplex and (n-1) dimensional Euclidean space.
    See https://en.wikipedia.org/wiki/Compositional_data#Linear_transformations for
    more information.
    Args:
        log_y: A log composition vector
        weight: A normalized vector of non-negative weights to use for the geometric
            mean. If `None`, a uniform reference distribution will be used.
        dim: The dimension along which to compute the geometric mean.
    Returns:
        The centered logratio vector.
    """
    # The geometric mean is simply the arithmetic mean in log space
    return log_y - _weighted_mean(log_y, weight, dim=dim).unsqueeze(dim)


def _weighted_mean(
    x: th.Tensor, weight: Optional[th.Tensor] = None, dim: int = -1
) -> th.Tensor:
    """Compute a weighted mean if `weight` is not `None`, else the unweighted mean."""
    if weight is None:
        return x.mean(dim=dim)

    # NOTE: `weight` is assumed to be non-negative and sum to 1.
    return x.mul(weight).sum(dim=dim)


def generate_aitchisons(prompts, lens, causal_basis):

    def subspace_ablation_hook(
        rs: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
        pos: list,
        subspace: Float[torch.Tensor, "d_model k"],
        sampled_rs: Float[torch.Tensor, "batch pos k"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:

        assert torch.allclose(torch.norm(subspace, dim=0), torch.ones_like(torch.norm(subspace, dim=0)), atol = 1e-8) 

        ablation = torch.zeros_like(rs[:,pos,:])
        delta = rs[:, pos, :] - sampled_rs[:,pos,:] # batch d_model

        P_u = subspace @ subspace.T #d_mod, d_mod
        rs[:, pos, :] = rs[:, pos, :] + (P_u @ delta.T).T  

        return rs + ablation


    # Hooked run
    model.reset_hooks(including_permanent=True)
    sample_idx = torch.randperm(len(prompts))

    _, pre_cache = model.run_with_cache(model.to_tokens(prompts.iloc[-1]))
    similarities = []
    for i, idx in enumerate(sample_idx):
        tokens = model.to_tokens(prompts[i])

        # Clean cache
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(tokens)
        clean_rs = torch.cat([clean_cache[f'blocks.{l}.hook_resid_post'] for l in range(n_layers)], 0) # [l p dm]
        clean_logits = clean_cache[f'ln_final.hook_normalized'] # [1 p dm]


        # Hooked cache
        hooked_lens = []
        hooked_logits = None

        for l in range(n_layers-1):

            hooked_lens_layer = []
            hooked_logit_layer = []
            
            for p in range(len(tokens)):
                model.reset_hooks(including_permanent = True)

                temp_ablation_fn = partial(subspace_ablation_hook, pos=p, subspace=causal_basis[i, l], sampled_rs=pre_cache[f'blocks.{l}.hook_resid_post'])
                model.blocks[l].hook_resid_post.add_hook(temp_ablation_fn) 

                with torch.no_grad():
                    _, hooked_cache = model.run_with_cache(tokens)
                hooked_lens_layer.append(hooked_cache[f'blocks.{l}.hook_resid_post'][:,p,:]) # [1 dm]
                hooked_logit_layer.append(hooked_cache[f'ln_final.hook_normalized'][:,p,:]) # [1 dm]

            #torch cat with layer
            hooked_lens.append(torch.cat(hooked_lens_layer, dim=0)[None, ...]) #[1 p dm]
            if hooked_logits is None:
                hooked_logits = torch.cat(hooked_logit_layer, dim=0)[None, ...] # [1 p dm]
            
        hooked_lens = torch.cat(hooked_lens, dim = 0) # [l p dm]
        
        # Compute Aitchison similarity
        simil = []

        response = torch.softmax(torch.log(model.unembed(hooked_logits).softmax(-1)) - torch.log(model.unembed(clean_logits).softmax(-1)), -1)[0] # [p dv]
        w = model.unembed(clean_logits).softmax(-1)[0]

        for l in range(n_layers-1):
            with torch.no_grad():
                stimuli = torch.softmax(torch.log(lens(hooked_lens[l], l).softmax(-1)) - torch.log(lens(clean_rs[l], l).softmax(-1)), -1) # [p dv]
                    
            simil.append((aitchison_similarity(torch.log(stimuli), torch.log(response), weight=w)).mean(-1)[None])

        pre_cache = clean_cache

        similarities.append(torch.cat(simil)[None, ...]) # l
    
    return torch.cat(similarities, dim = 0) # batch l