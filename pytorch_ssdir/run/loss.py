"""SSDIR loss."""
from typing import Dict

import pyro.poutine as poutine


def per_site_loss(model, guide, *args, **kwargs) -> Dict[str, float]:
    """Calculate loss fn for each site."""
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(
        *args, **kwargs
    )

    losses: Dict[str, float] = {}
    for trace in [model_trace, guide_trace]:
        for site in trace.nodes.values():
            if site["type"] == "sample" and "data" not in site["name"]:
                name = site["name"]
                elbo = losses.get(name, 0.0)
                losses[name] = elbo - site["fn"].log_prob(site["value"]).sum()

    return losses
