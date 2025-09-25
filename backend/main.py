import modal

app = modal.App('jukebox')

image = (
    modal.Image.debian_slim()
    .apt_install('git')
    .pip_install_from_requirements('requirements.txt')
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step", "cd /tmp/ACE-Step && pip install ."])
    .env({"HF_HOME": "/.cache/huggingface"})
)