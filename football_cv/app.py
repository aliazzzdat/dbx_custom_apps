import gradio as gr
import cv2
import numpy as np
import yt_dlp
import os
from PIL import Image
import io
import base64
import requests
import pandas as pd
import json

DATABRICKS_HOST = os.getenv('DATABRICKS_HOST')
DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN')
DATABRICKS_ENDPOINT = os.getenv('DATABRICKS_ENDPOINT')

MODES = [
    "PITCH_DETECTION",
    "PLAYER_DETECTION",
    "BALL_DETECTION",
    "PLAYER_TRACKING",
    "TEAM_CLASSIFICATION",
    "RADAR"
]

FRAMES_LIMIT = 5

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset, params):
    url = f"{DATABRICKS_HOST}/serving-endpoints/{DATABRICKS_ENDPOINT}/invocations"
    headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    ds_dict.update({"params": params})
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def decode_image(encoded_string):
    decoded_data = base64.b64decode(encoded_string)
    image = Image.open(io.BytesIO(decoded_data))
    return image

def get_sample_videos():
    return [f for f in os.listdir("data") if (f.endswith(".mp4") & ('downloaded_video.mp4' not in f))]

def extract_video_frames(video):
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_frames(video_input, image_input, sample_video_input, url_input):
    if video_input is not None:
        frames = extract_video_frames(video_input)
    elif image_input is not None:
        if isinstance(image_input, np.ndarray):
            frames = [image_input]
        elif isinstance(image_input, str):
            image = cv2.imread(image_input)
            frames = [image]
    elif sample_video_input is not None:
        frames = extract_video_frames(f"data/{sample_video_input}")
    elif url_input is not None:
        ydl_opts = {
            'outtmpl': "data/downloaded_video.mp4",
            'format': 'best[ext=mp4][filesize<30M]',
        }
        try:
            ydl = yt_dlp.YoutubeDL(ydl_opts)
            ydl.download([url_input])
        #except yt_dlp.utils.DownloadError:
        #     print("Error: No suitable format found under 30MB")
        except Exception as e:
            print(e)
        frames = extract_video_frames("data/downloaded_video.mp4")
    frames = [(frame, f"Frame {i}") for i, frame in enumerate(frames)]
    return frames, gr.Slider(minimum=0, maximum=len(frames)-1, step=1, value=0)

    
def apply_cv(gallery, frame_index, cv_type):
    if not gallery:
        return None
    selected_frames = gallery[int(frame_index):int(frame_index)+FRAMES_LIMIT]
    frames_label = [frame[1] for frame in selected_frames]
    selected_frames = [frame[0] for frame in selected_frames]
    
    df_input = pd.DataFrame({
        'encoded_image': [encode_image(frame) for frame in selected_frames]
    })
    params = {'mode': cv_type}
    df_output = score_model(df_input, params)
    df_output = pd.DataFrame(df_output['predictions'])
    cv_frames = [decode_image(encoded_image) for encoded_image in df_output['encoded_image']]

    result_frames = list(zip(cv_frames, frames_label))
    return result_frames


with gr.Blocks(title="Football Computer Vision") as demo:
    gr.Markdown(
    """
    # Football Computer Vision

    Upload a video or an image, provide a YouTube URL, or select a sample video to extract frames. Then select a mode and click on "Add cv".

    Please note this app has been made for demonstration purposes only, the models have been deployed on a CPU endpoint which may delays the processing and limits the number of frames that can be processed (5 frames maximum). 

    Computer vision models should run on on-edge with the app using GPU device for real-time processing.

    Please note PLAYER_TRACKING, TEAM_CLASSIFICATION and RADAR require a video input (at least 3 frames).

    [Access the model in Unity Catalog](https://e2-demo-field-eng.cloud.databricks.com/explore/data/models/ali_azzouz/football/football_cv?o=1444828305810485)
    """)
    
    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        image_input = gr.Image(label="Upload Image")
        sample_video_input = gr.Dropdown(choices=get_sample_videos(), label="Select Sample Video", value=get_sample_videos()[0])
        url_input = gr.Text(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=_25ljC5d5hI")
        submit_button = gr.Button("Extract Frames")
        clear_button = gr.Button("Clear")
    
    with gr.Row():
        output_gallery = gr.Gallery(label="Extracted Frames", preview=True, show_label=True, interactive=False)
    
    with gr.Row():
        frame_selector = gr.Slider(minimum=0, maximum=0, step=1, label="Selected Frame")
        cv_type = gr.Dropdown(choices=MODES, label="CV Type", value="BALL_DETECTION")
        add_cv_button = gr.Button("Add cv")
    
    with gr.Row():
        output_cv_gallery = gr.Gallery(label="Output Frames With CV", preview=True, show_label=True, interactive=False)
    
    def clear_inputs():
        return {
            video_input: None,
            image_input: None,
            sample_video_input: get_sample_videos()[0],
            url_input: "",
            output_gallery: None,
            frame_selector: gr.Slider(minimum=0, maximum=0, step=1, value=0),
            cv_type: "BALL_DETECTION",
            output_cv_gallery: None
        }

    clear_button.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[video_input, image_input, sample_video_input, url_input, output_gallery, frame_selector, cv_type, output_cv_gallery]
    )
    
    def select_fn(data: gr.SelectData):
        return data.index#, data.value
    
    output_gallery.select(select_fn, None, frame_selector)

    submit_button.click(
        fn=extract_frames,
        inputs=[video_input, image_input, sample_video_input, url_input],
        outputs=[output_gallery, frame_selector]
    )
    
    add_cv_button.click(
        fn=apply_cv,
        inputs=[output_gallery, frame_selector, cv_type],
        outputs=output_cv_gallery
    )

demo.launch()
