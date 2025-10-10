import modal
import os
import uuid
import base64
from pydantic import BaseModel
import requests

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


class GenerateMusicResponse(BaseModel):
    audio_data: str


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secret],
    scaledown_window=15
)
class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from diffusers import AutoPipelineForText2Image
        import torch

     # Music genertion model
        self.music_model = ACEStepPipeline(checkpoint_dir="/models", dtype="bfloat16",
                                           torch_compile=False, cpu_offload=False, overlapped_decode=False)

        # Large Language Model
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )

    # Stable diffusion model (thumbnails)
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", cache_dir="/.cache/huggingface")
        self.image_pipe.to("cuda")

    @modal.fastapi_endpoint(method="POST")
    def generate(self) -> GenerateMusicResponse:
        output_dir = "tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

        self.music_model(
            prompt="hip hop, alternative/indie, energetic, spoken word, young adult, male, rap music",
            lyrics="""[Intro]\nYou ready?! Let's go!\nYeah, for those of you that want to know what we're all about\nIt's like this y'all (c'mon!)\n\n[Chorus]\nThis is ten percent luck, twenty percent skill\nFifteen percent concentrated power of will\nFive percent pleasure, fifty percent pain\nAnd a hundred percent reason to remember the name!\n\n[Verse 1]\nMike, he doesn't need his name up in lights\nHe just wants to be heard whether it's the beat or the mic\nHe feels so unlike everybody else, alone\nIn spite of the fact that some people still think that they know him\nBut fck 'em, he knows the code, it's not about the salary\nIt's all about reality and making some noise\nMakin' the story, makin' sure his clique stays up\nThat means when he puts it down, Tak's pickin' it up, let's go!\n\n[Verse 2]\nWho the hell is he anyway? He never really talks much\nNever concerned with status but still leavin' 'em star-struck\nHumbled through opportunities given despite the fact\nThat many misjudge him because he makes a livin' from writin' raps\nPut it together himself, now the picture connects\nNever askin' for someone's help, or to get some respect\nHe's only focused on what he wrote, his will is beyond reach\nAnd now it all unfolds, the skill of an artist\n\n[Verse 3]\nThis is twenty percent skill, eighty percent beer\nBe a hundred percent clear 'cause Ryu is ill\nWho would've thought that he'd be the one to set the west in flames?\nAnd I heard him wreck it with The Crystal Method, "Name of the Game"\nCame back, dropped "Megadef, " took 'em to church\nI like "Bleach, " man, Ryu had the stupidest verse\nThis dude is the truth, now everybody givin' him guest spots\nHis stock's through the roof, I heard he's fckin' with S. Dot!\n\n[Chorus]\nThis is ten percent luck, twenty percent skill\nFifteen percent concentrated power of will\nFive percent pleasure, fifty percent pain\nAnd a hundred percent reason to remember the name!\n\n[Verse 4]\nThey call him Ryu the sick, and he's spittin' fire with Mike\nGot him out the dryer, he's hot, found him in Fort Minor with Tak\nWhat a fckin' nihilist, porcupine, he's a prick, he's a cck\nThe type of woman want to be with and rappers hope he get shot\nEight years in the makin', patiently waitin' to blow\nNow the record with Shinoda's takin' over the globe\nHe's got a partner in crime, his sht is equally dope\nYou won't believe the kind of sht that comes out of this kid's throat\n\n[Verse 5]\nHe's not your everyday on the block\nHe knows how to work with what he's got\nMakin' his way to the top\nHe often gets a comment on his name\nPeople keep askin' him if it was given at birth\nOr does it stand for an acronym? No\nHe's livin' proof, got him rockin' the booth\nHe'll get you buzzin' quicker than a shot of vodka with juice\nHim and his crew are known around as one of the best\nDedicated to what they doin', give a hundred percent\n\n[Verse 6]\nForget Mike, nobody really knows how or why he works so hard\nIt seems like he's never got time\nBecause he writes every note and he writes every line\nAnd I've seen him at work when that light goes on in his mind\nIt's like a design is written in his head every time\nBefore he even touches a key or speaks in a rhyme\nAnd those motherf*ckers he runs with, the kids that he signed?\nRidiculous, without even tryin', how do they do it?!\n\n[Chorus]\nThis is ten percent luck, twenty percent skill\nFifteen percent concentrated power of will\nFive percent pleasure, fifty percent pain\nAnd a hundred percent reason to remember the name!""",
            audio_duration=180,
            infer_step=60,
            guidance_scale=15,
            save_path=output_path
        )

        with open(output_path, "rb") as file:
            audio_bytes = file.read()

        audiob64 = base64.b64encode(audio_bytes).decode("utf-8")
        os.remove(output_path)
        return GenerateMusicResponse(audio_data=audiob64)


@app.local_entrypoint()
def main():
    server = MusicGenServer()
    endpoint_url = server.generate.get_web_url()

    response = requests.post(endpoint_url)
    response.raise_for_status()
    result = GenerateMusicResponse(**response.json())

    audio_bytes = base64.b64decode(result.audio_data)
    output_filename = 'generated.wav'
    with open(output_filename, 'wb') as f:
        f.write(audio_bytes)
