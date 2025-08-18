import asyncio
from starlette.requests import Request
from ray import serve
from ray.serve._private.http_util import ASGIAppReplicaWrapper
from fastvideo import VideoGenerator
import base64
import io
import imageio
import uuid
import os
import gradio as gr

example_prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

output_dir = "gradio_videos"
os.makedirs(output_dir, exist_ok=True)


def gradio_builder(generator: serve.handle.DeploymentHandle):
    def query_model(prompt):

        async def run_query_model(prompt):
            video_base64 = await generator.generate.remote(prompt)
            return video_base64

        video_base64 = asyncio.run(run_query_model(prompt))
        video_bytes = base64.b64decode(video_base64)
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


@serve.deployment
class GradioServer(ASGIAppReplicaWrapper):
    """User-facing class that wraps a Gradio App in a Serve Deployment."""

    def __init__(self, generator: serve.handle.DeploymentHandle):
        self.generator = generator
        ui = gradio_builder(generator)
        super().__init__(gr.routes.App.create_app(ui))


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1, "memory": 50 * 10**9, "accelerator_type": "L4"})
class GenerateVideo:
    def __init__(self):
        # Create a video generator with a pre-trained model
        self.generator = VideoGenerator.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            num_gpus=1,  # Adjust based on your hardware
        )

    def generate(self, prompt: str) -> bytes:
        # Generate the video
        video = self.generator.generate_video(
            prompt,
            num_inference_steps=5, # JUST TO SPEED UP TESTING
            return_frames=True,  # Also return frames from this call (defaults to False)
            # output_path="my_videos/",  # Controls where videos are saved
            # save_video=True
        )

        buffer = io.BytesIO()
        imageio.mimsave(buffer, video, fps=16, format="mp4")
        buffer.seek(0)
        video_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return video_base64

    async def __call__(self, http_request: Request) -> bytes:
        prompt: str = await http_request.json()
        print(prompt)
        return self.generate(prompt)


app = GradioServer.bind(GenerateVideo.bind())
