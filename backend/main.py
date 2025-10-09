import modal

app = modal.App("jukebox")

image = (
    modal.Image.debian_slim()
    .apt_install('git')
    .pip_install_from_requirements('requirements.txt')
    .run_commands(['git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step', 'cd /tmp/ACE-Step && pip install .'])
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name(
    "ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secret = modal.Secret.from_name("music-gen-secret")


@app.cls(
    image=image,
    gpu="L405",
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secret],
    scaledown_window=15
)
class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoTokenizer, AutoModelForCasualLM
        from diffusers import AutoPipelineForText2Image
        import torch

     # Music genertion model
        self.music_model = ACEStepPipeline(checkpoint_dir="/models", dtype="bfloat16",
                                           torch_compile=False, cpu_offload=False, overlapped_decode=False)

        # Large Language Model
        model_id = "Qwen/Qwen2-72B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm_model = AutoModelForCasualLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )

    # Stable diffusion model (thumbnails)
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", cache_dir="/.cache/huggingface")
        self.image_pipe.to("cuda")


@app.local_entrypoint()
def main():
    pass
