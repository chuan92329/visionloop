"""Load a Hugging Face model from the local cache without asking the Hub first.

By default transformers checks huggingface.co for a newer version on every load, even when the model
is already downloaded. Without a connection that check only gives up after a network timeout
(more than a minute), so click-to-segment and the other tools seemed to do nothing. Models that are
already downloaded are loaded straight from the cache; only a missing model goes to the Hub.

Imported by the helper scripts, which run in the model environment.
"""

from __future__ import annotations


def from_pretrained(cls, model_id: str, **kwargs):
    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:  # not downloaded yet (or incomplete): download it
        return cls.from_pretrained(model_id, **kwargs)
