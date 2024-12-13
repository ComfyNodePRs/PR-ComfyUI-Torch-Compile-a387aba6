import torch

# Based on https://github.com/kijai/ComfyUI-KJNodes/blob/8c590fd5a023ee14b5617347567752bf62ea4cd6/nodes/model_optimization_nodes.py#L321


class TorchCompileLoadVAE:
    CATEGORY = "torch-compile"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "compile"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "backend": (["inductor", "cudagraphs"],),
                "fullgraph": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable full graph mode"},
                ),
                "mode": (
                    [
                        "default",
                        "max-autotune",
                        "max-autotune-no-cudagraphs",
                        "reduce-overhead",
                    ],
                    {"default": "default"},
                ),
                "compile_encoder": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Compile encoder"},
                ),
                "compile_decoder": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Compile decoder"},
                ),
            }
        }

    def compile(
        self,
        vae,
        backend: str,
        mode: str,
        fullgraph: bool,
        compile_encoder: bool,
        compile_decoder: bool,
    ):
        if compile_encoder:
            encoder_name = "encoder"
            if hasattr(vae.first_stage_model, "taesd_encoder"):
                encoder_name = "taesd_encoder"

            setattr(
                vae.first_stage_model,
                encoder_name,
                torch.compile(
                    getattr(vae.first_stage_model, encoder_name),
                    mode=mode,
                    fullgraph=fullgraph,
                    backend=backend,
                ),
            )
        if compile_decoder:
            decoder_name = "decoder"
            if hasattr(vae.first_stage_model, "taesd_decoder"):
                decoder_name = "taesd_decoder"

            setattr(
                vae.first_stage_model,
                decoder_name,
                torch.compile(
                    getattr(vae.first_stage_model, decoder_name),
                    mode=mode,
                    fullgraph=fullgraph,
                    backend=backend,
                ),
            )
        return (vae,)
