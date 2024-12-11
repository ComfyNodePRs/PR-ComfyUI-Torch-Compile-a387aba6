from .nodes.torch_compile_vae import TorchCompileVAE
from .nodes.torch_compile_controlnet import TorchCompileControlNet

NODE_CLASS_MAPPINGS = {
    "TorchCompileVAE": TorchCompileVAE,
    "TorchCompileControlNet": TorchCompileControlNet,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
