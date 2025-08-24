# Process images

This example uses Ray Data to process the [ReLAION-2B](https://huggingface.co/datasets/laion/relaion2B-en-research-safe) image dataset, which consists of over 2 billion rows. Each row consists of an image URL along with various metadata include a caption and image dimensions.

## Install the Anyscale CLI

```
anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN
```