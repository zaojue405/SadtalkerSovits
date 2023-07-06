import os, sys
import tempfile
import gradio as gr
from src.gradio_demo import SadTalker  
import json
import os
import subprocess
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import DEFAULT_MODEL, get_model
from huggingface_hub import hf_hub_download, list_repo_files

from so_vits_svc_fork.hparams import HParams
from so_vits_svc_fork.inference.core import Svc


###################################################################
# REPLACE THESE VALUES TO CHANGE THE MODEL REPO/CKPT NAME/SETTINGS
###################################################################
# The Hugging Face Hub repo ID - 在这里修改repo_id，可替换成任何已经训练好的模型！
speakers=['Trump','Sun','MasterMa','SunxiaoChuan']
speakers_dict = {"Trump":"Nardicality/so-vits-svc-4.0-models","Sun":"YeQing/sunyanzi",
        "MasterMa":"zaojue405/MasterMa","SunxiaoChuan":"zaojue405/sunxiaochuan"}
# repo_id = "Nardicality/so-vits-svc-4.0-models"

# If None, Uses latest ckpt in the repo
# ckpt_name = None

# If None, Uses "kmeans.pt" if it exists in the repo
cluster_model_name = None

# Set the default f0 type to use - use the one it was trained on.
# The default for so-vits-svc-fork is "dio".
# Options: "crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
default_f0_method = "crepe"

# The default ratio of cluster inference to SVC inference.
# If cluster_model_name is not found in the repo, this is set to 0.
default_cluster_infer_ratio = 0.5

# Limit on duration of audio at inference time. increase if you can
# In this parent app, we set the limit with an env var to 30 seconds
# If you didnt set env var + you go OOM try changing 9e9 to <=300ish
duration_limit = int(os.environ.get("MAX_DURATION_SECONDS", 9e9))
###################################################################

# Figure out the latest generator by taking highest value one.
# Ex. if the repo has: G_0.pth, G_100.pth, G_200.pth, we'd use G_200.pth
# ckpt_name='Trump18.5k/G_18500.pth'
ckpt_name_dict = {"Trump":"Trump18.5k/G_18500.pth","Sun":"G_27200.pth","MasterMa":"G_392800.pth",
"SunxiaoChuan":"G_2520.pth"}
config_dict =  {"Trump":"Trump18.5k/config.json","Sun":"config.json","MasterMa":"config2.json",
"SunxiaoChuan":"config.json"}
# generator_path = hf_hub_download(repo_id, ckpt_name)
# config_path = hf_hub_download(repo_id, "Trump18.5k/config.json")
# hparams = HParams(**json.loads(Path(config_path).read_text()))
# speakers = list(hparams.spk.keys())
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = Svc(net_g_path=generator_path, config_path=config_path, device=device, cluster_model_path=None)
# demucs_model = get_model(DEFAULT_MODEL)
model = None
predict_speakers = None
speaker_present = None
def download_model(speaker):
    global model
    
    ckpt_name = ckpt_name_dict[speaker]
    
    model_dir = r".\models"  # 模型存放的目录路径
    generator_path = os.path.join(model_dir, ckpt_name)
    config_path = os.path.join(model_dir, config_dict[speaker])
    print(generator_path)
    print(config_path)
    
    # Check if the model files already exist locally
    if os.path.exists(generator_path) and os.path.exists(config_path):
        pass  # 文件已经存在，不需要下载
    else:
        # 下载模型文件到指定的目录
        repo_id = speakers_dict[speaker]
        generator_path = hf_hub_download(repo_id, ckpt_name, destination=model_dir)
        config_path = hf_hub_download(repo_id, config_dict[speaker], destination=model_dir)
    
    hparams = HParams(**json.loads(Path(config_path).read_text()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speakers = list(hparams.spk.keys())
    model = Svc(net_g_path=generator_path, config_path=config_path, device=device, cluster_model_path=None)
    
    return speakers

def predict(
    speaker,
    audio,
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    f0_method: str = "crepe",
    db_thresh: int = -40,
    pad_seconds: float = 0.5,
    chunk_seconds: float = 0.5,
    absolute_thresh: bool = False,
):
    global speaker_present
    global predict_speakers
    print(speaker)
    if not model:
        speaker_present = speaker
        predict_speakers= download_model(speaker)
    if speaker != speaker_present:
        speaker_present = speaker
        predict_speakers= download_model(speaker)
    audio, _ = librosa.load(audio, sr=model.target_sample, duration=duration_limit)
    audio = model.infer_silence(
        audio.astype(np.float32),
        speaker=predict_speakers[0],
        transpose=transpose,
        auto_predict_f0=auto_predict_f0,
        cluster_infer_ratio=cluster_infer_ratio,
        noise_scale=noise_scale,
        f0_method=f0_method,
        db_thresh=db_thresh,
        pad_seconds=pad_seconds,
        chunk_seconds=chunk_seconds,
        absolute_thresh=absolute_thresh,
    )
    return model.target_sample, audio






def get_source_image(image):   
        return image

def sadtalker_demo():

    sad_talker = SadTalker(lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")
        
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", source="upload", type="filepath").style(height=256,width=256)
               
 
                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload OR TTS'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", source="upload", type="filepath")
                with gr.Tabs(elem_id="sadtalker_source_audio_mic"):
                    with gr.TabItem('From microphone'):
                        with gr.Row():
                            speaker=gr.Dropdown(speakers, value=speakers[0], label="Target Speaker")

                        with gr.Row():
                            source_audio = gr.Audio(type="filepath", source="microphone", label="Source Audio")
                            microphone = gr.Button('convert',elem_id='sadtalker_audio_record')
                            print(speaker)
                            microphone.click(fn=predict,inputs=[speaker,source_audio],outputs=[driven_audio])

                       

            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            preprocess_type = gr.Radio(['crop','resize','full'], value='crop', label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(label="w/ Still Mode (fewer hand motion, works with preprocess `full`)")
                            enhancer = gr.Checkbox(label="w/ GFPGAN as Face enhancer")
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)



       

    return sadtalker_interface
 

if __name__ == "__main__":

    demo = sadtalker_demo()
    demo.launch(share=True,server_name='0.0.0.0',server_port=80,ssl_verify=False)


