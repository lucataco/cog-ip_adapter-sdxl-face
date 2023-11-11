# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import sys
sys.path.extend(['/IP-Adapter'])
import torch
import shutil
from PIL import Image
from typing import List
from ip_adapter import IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/IP-Adapter/models/image_encoder/"
ip_ckpt = "/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin"
device = "cuda"
MODEL_CACHE = "model-cache"

def load_image(path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # load SDXL pipeline
        self.pipe = StableDiffusionXLCustomPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
            cache_dir=MODEL_CACHE,
        )

    def predict(
        self,
        image: Path = Input(
             description="Input face image"
        ),
        prompt: str = Input(
            description="Prompt (leave blank for image variations)",
            default=""
        ),
        negative_prompt: str = Input(
            description="Negative Prompt",
            default=""
        ),
        scale: float = Input(
            description="Scale (influence of input image on generation)",
            ge=0.0,
            le=1.0,
            default=0.6
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = Image.open(image)
        image.resize((224, 224))

        # load ip-adapter
        ip_model = IPAdapterPlusXL(self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

        images = ip_model.generate(
            pil_image=image,
            num_samples=num_outputs,
            num_inference_steps=num_inference_steps,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            scale=scale
        )

        output_paths = []
        for i, _ in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            images[i].save(output_path)
            output_paths.append(Path(output_path))
            
        return output_paths
