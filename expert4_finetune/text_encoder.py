"""LAION-CLAP RoBERTa tokens and pretrained text projection."""
from pathlib import Path

import torch
from torch import nn


def verify_clap_projection_state(state):
    shapes = {name.removeprefix('module.'): tuple(value.shape) for name, value in state.items()
              if 'text_projection' in name and hasattr(value, 'shape')}
    expected = {'text_projection.0.weight': (512, 768), 'text_projection.2.weight': (512, 512)}
    for key, shape in expected.items():
        if shapes.get(key) != shape:
            raise ValueError(f'LAION-CLAP checkpoint needs pretrained {key} with shape {shape}; found {shapes.get(key)}')


def verify_clap_roberta_state(state):
    """Require the canonical RoBERTa-base embedding geometry in the CLAP file."""
    expected = {
        'text_branch.embeddings.word_embeddings.weight': (50265, 768),
        'text_branch.embeddings.position_embeddings.weight': (514, 768),
        'text_branch.embeddings.token_type_embeddings.weight': (1, 768),
    }
    for key, shape in expected.items():
        tensor = state.get(key, state.get(f'module.{key}'))
        actual = tuple(tensor.shape) if tensor is not None else None
        if actual != shape:
            raise ValueError(f'LAION-CLAP checkpoint needs {key} with shape {shape}; found {actual}')


def roberta_base_config():
    """The public roberta-base config; random weights are replaced by CLAP state."""
    from transformers import RobertaConfig
    return RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12,
                         num_attention_heads=12, intermediate_size=3072,
                         max_position_embeddings=514, type_vocab_size=1)


class TextEncoder(nn.Module):
    def __init__(self, branch, projection, tokenizer):
        super().__init__()
        self.branch, self.projection, self.tokenizer = branch, projection, tokenizer
        if branch.config.model_type != 'roberta' or branch.config.hidden_size != 768:
            raise ValueError('Expected the 768-dimensional LAION-CLAP RoBERTa branch')
        if len(branch.encoder.layer) != 12:
            raise ValueError('Expected twelve RoBERTa transformer layers')
        for param in branch.parameters():
            param.requires_grad = False
        for layer in branch.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in projection.parameters():
            param.requires_grad = True
        with torch.no_grad():
            try:
                result = projection(torch.zeros(2, 3, 768))
            except Exception as exc:
                raise ValueError('CLAP text_projection cannot project token sequences') from exc
        if tuple(result.shape) != (2, 3, 512):
            raise ValueError(f'CLAP text_projection must map [B,T,768] to [B,T,512], got {tuple(result.shape)}')

    @classmethod
    def load_pretrained(cls, checkpoint):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f'LAION-CLAP checkpoint not found: {checkpoint}')
        try:
            import laion_clap
            from transformers import RobertaModel
            # This user-supplied checkpoint contains a NumPy scalar and cannot be
            # loaded with weights_only=True in current PyTorch. Treat it as trusted.
            try:
                raw = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
            except (RuntimeError, ValueError) as exc:
                if 'mmap' not in str(exc).lower():
                    raise
                raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
            state = raw.get('state_dict', raw.get('model', raw))
            verify_clap_projection_state(state)
            verify_clap_roberta_state(state)
            # LAION-CLAP constructs roberta-base with from_pretrained before
            # loading its own checkpoint. Build the identical base architecture
            # locally, then load all text weights from the supplied CLAP file.
            marker = object()
            original = RobertaModel.__dict__.get('from_pretrained', marker)
            def local_roberta(cls, name, *args, **kwargs):
                if name != 'roberta-base':
                    raise ValueError(f'Unexpected CLAP text model: {name}')
                return cls(roberta_base_config())
            RobertaModel.from_pretrained = classmethod(local_roberta)
            try:
                clap = laion_clap.CLAP_Module(enable_fusion=False, device='cpu')
            finally:
                if original is marker:
                    delattr(RobertaModel, 'from_pretrained')
                else:
                    RobertaModel.from_pretrained = original
            available = {name.removeprefix('module.') for name in state}
            missing = {f'text_branch.{name}' for name, _ in clap.model.text_branch.named_parameters()} - available
            if missing:
                raise ValueError(f'CLAP checkpoint lacks RoBERTa parameters: {sorted(missing)[:8]}')
            cleaned = {name.removeprefix('module.'): value for name, value in state.items()}
            missing_keys, _ = clap.model.load_state_dict(cleaned, strict=False)
            required_missing = [name for name in missing_keys if name.startswith(('text_branch.', 'text_projection.'))]
            if required_missing:
                raise ValueError(f'CLAP text parameters were not loaded: {required_missing[:8]}')
            del raw, state, cleaned
            return cls(clap.model.text_branch, clap.model.text_projection, clap.tokenize)
        except Exception as exc:
            raise RuntimeError(f'Could not load trusted LAION-CLAP checkpoint {checkpoint}: {exc}') from exc

    def forward(self, captions):
        if not captions or any(not isinstance(s, str) or not s.strip() for s in captions):
            raise ValueError('Captions must be nonempty text strings')
        device = next(self.branch.parameters()).device
        tokens = self.tokenizer(captions, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        ids = tokens['input_ids'].to(device)
        attention = tokens['attention_mask'].to(device)
        hidden = self.branch(input_ids=ids, attention_mask=attention, return_dict=True).last_hidden_state
        # The repository's projected-token helper removes the first token and its mask.
        features = self.projection(hidden[:, 1:, :]).float()
        mask = attention[:, 1:].eq(0)
        if features.shape[:2] != mask.shape or features.shape[-1] != 512 or not torch.isfinite(features).all() or mask.all(dim=1).any():
            raise ValueError('Invalid CLAP projected tokens or padding mask')
        return features, mask
