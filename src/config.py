# Handles config loading for Wiki RAG

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'


def load_config(path=CONFIG_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Ensure some expected sections exist to avoid KeyError later
    if 'models' not in cfg:
        cfg['models'] = {}
    if 'llm' not in cfg['models']:
        cfg['models']['llm'] = {
            'enabled': True,
            'provider': 'gpt4all',
            'model_path': 'models/gpt4all-model.bin',
            'model_url': '',
            'model_sha256': '',
            'auto_download': True,
            'prompt_for_download': True,
            'prompt_template_path': 'prompts/german_default.txt',
            'max_tokens': 512,
            'temperature': 0.0,
            'n_ctx': 2048,
        }

    return cfg

