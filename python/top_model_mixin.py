from typing import Optional

from .lora_manager import LoraConfig
from .mapping import Mapping
from .plugin.plugin import PluginConfig

class TopModelMixin:

    def __init__(self) -> None:
        pass

    @classmethod
    def from_hugging_face(cls,
                          hf_model_dir: str,
                          dtype: Optional[str] = 'float16',
                          mapping: Optional[Mapping] = None,
                          **kwargs):
        raise NotImplementedError("Subclass shall override this")

    def use_lora(self, lora_config: LoraConfig):
        raise NotImplementedError("Subclass shall override this")

    def use_prompt_tuning(self, max_prompt_embedding_table_size: str,
                          prompt_table_path: str):
        raise NotImplementedError

    def default_plugin_config(self, **kwargs) -> PluginConfig:
        return PluginConfig.from_dict(kwargs)
