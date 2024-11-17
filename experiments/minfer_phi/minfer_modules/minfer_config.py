import os
from minference.models_patch import MInference  
from minference.minference_configuration import MInferenceConfig

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
EXPR_MODEL2PATH = {
    "microsoft/Phi-3-mini-4k-instruct": os.path.join(
        CONFIG_DIR, "Phi-3-mini-4k-instruct-LongRoPE-128k.json"
    ),
}

class ExprMInferConfig(MInferenceConfig):
    def update_config_path(self, config_path: str, model_name: str):
        if model_name in EXPR_MODEL2PATH:
            return EXPR_MODEL2PATH[model_name]

        return super().update_config_path(config_path, model_name)

class ExprMInference(MInference):
    def __init__(
        self,
        attn_type: str = "minference",
        model_name: str = None,
        config_path: str = None,
        starting_layer: int = -1,
        kv_cache_cpu: bool = False,
        use_snapkv: bool = False,
        is_search: bool = False,
        attn_kwargs: dict = {},
        **kwargs,
    ):
        super(MInference, self).__init__()
        self.config = ExprMInferConfig(
            attn_type=attn_type,
            model_name=model_name,
            config_path=config_path,
            starting_layer=starting_layer,
            kv_cache_cpu=kv_cache_cpu,
            use_snapkv=use_snapkv,
            is_search=is_search,
            attn_kwargs=attn_kwargs,
            **kwargs,
        )

