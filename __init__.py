from .nodes.torch_compile_vae import TorchCompileLoadVAE
from .nodes.torch_compile_controlnet import TorchCompileLoadControlNet

NODE_CLASS_MAPPINGS = {
    "TorchCompileLoadVAE": TorchCompileLoadVAE,
    "TorchCompileLoadControlNet": TorchCompileLoadControlNet,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
