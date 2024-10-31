# -*- coding:UTF-8 -*-
# !/usr/bin/env python
import spaces
import numpy as np
import gradio as gr
import gradio.exceptions
import roop.globals
from roop.core import (
    start,
    decode_execution_providers,
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
import os
from PIL import Image
import uuid
import onnxruntime as ort
import cv2
from roop.face_analyser import get_one_face

@spaces.GPU
def swap_face(source_file, target_file, doFaceEnhancer):
    session_id = str(uuid.uuid4())  # Tạo một UUID duy nhất cho mỗi phiên làm việc
    session_dir = f"temp/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    source_path = os.path.join(session_dir, "input.jpg")
    target_path = os.path.join(session_dir, "target.jpg")

    source_image = Image.fromarray(source_file)
    source_image.save(source_path)
    target_image = Image.fromarray(target_file)
    target_image.save(target_path)

    print("source_path: ", source_path)
    print("target_path: ", target_path)

    # Check if a face is detected in the source image
    source_face = get_one_face(cv2.imread(source_path))
    if source_face is None:
        raise gradio.exceptions.Error("No face in source path detected.")

    # Check if a face is detected in the target image
    target_face = get_one_face(cv2.imread(target_path))
    if target_face is None:
        raise gradio.exceptions.Error("No face in target path detected.")

    output_path = os.path.join(session_dir, "output.jpg")
    normalized_output_path = normalize_output_path(source_path, target_path, output_path)

    frame_processors = ["face_swapper", "face_enhancer"] if doFaceEnhancer else ["face_swapper"]


    for frame_processor in get_frame_processors_modules(frame_processors):
        if not frame_processor.pre_check():
            print(f"Pre-check failed for {frame_processor}")
            raise gradio.exceptions.Error(f"Pre-check failed for {frame_processor}")

    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    roop.globals.output_path = normalized_output_path
    roop.globals.frame_processors = frame_processors
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_audio = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = False
    roop.globals.video_encoder = "libx264"
    roop.globals.video_quality = 18
    roop.globals.execution_providers = ["CUDAExecutionProvider"]
    roop.globals.reference_face_position = 0
    roop.globals.similar_face_distance = 0.6
    roop.globals.max_memory = 60
    roop.globals.execution_threads = 50
    
    start()
    return normalized_output_path

app = gr.Interface(
    fn=swap_face, 
    inputs=[
        gr.Image(), 
        gr.Image(), 
        gr.Checkbox(label="Face Enhancer?", info="Do face enhancement?")
    ], 
    outputs="image"
)
app.launch()