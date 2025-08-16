from starlette.requests import Request
import ray
from ray import serve
from fastvideo import VideoGenerator
from typing import Any, Dict
import base64
import io
import imageio

@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
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


app = GenerateVideo.bind()

