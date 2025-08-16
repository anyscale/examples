import gradio as gr
import requests
import base64
import os
import uuid


base_url = "http://127.0.0.1:8000/"  # Fill this in.

example_prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."


output_dir = "gradio_videos"
os.makedirs(output_dir, exist_ok=True)


def gradio_summarizer_builder():

    def query_model(prompt):
        response = requests.post(base_url, json=prompt)
        video_data = response.text

        video_bytes = base64.b64decode(video_data)
        video_filename = f"{uuid.uuid4()}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        
        return video_path

    with gr.Blocks() as ui:
        prompt = gr.Text(
            label="Prompt",
            value=example_prompt,
            show_label=False,
            max_lines=3,
            placeholder="Describe your scene...",
            container=False,
            lines=3,
            autofocus=True,
        )
        run_button = gr.Button("Run", variant="primary", size="lg")
        result = gr.Video(
            label="Generated Video", 
            show_label=True,
            height=466,
            width=600,
            container=True,
            elem_classes="video-component")

        run_button.click(
            fn=query_model,
            inputs=[prompt],
            outputs=[result],
        )
    return ui

if __name__ == "__main__":
    grad = gradio_summarizer_builder()
    grad.queue(max_size=20).launch(server_name="0.0.0.0", server_port=8888)
