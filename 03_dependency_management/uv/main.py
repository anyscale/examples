import emoji
import ray


@ray.remote
def f():
    return emoji.emojize("Python is :thumbs_up:")


print(ray.get(f.remote()))