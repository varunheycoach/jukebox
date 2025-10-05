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

model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secret = modal.Secret.from_name("music-gen-secret")

@app.cls(
    image=image,
    gpu="L405",
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume}
    secrets=[music_gen_secret]
    scaledown_window=15
)
class MusicGenServer:
    @modal.enter()
    def load_model(self):
        pass
@app.local_entrypoint()
def main():
    pass