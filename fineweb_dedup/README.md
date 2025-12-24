# Large-Scale Text Data Processing with Data-Juicer

This example demonstrates how to build a scalable text data processing pipeline using [Data-Juicer](https://github.com/modelscope/data-juicer) and Ray Data on Anyscale. We process the [FineWeb-edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a high-quality educational web text corpus.

You'll need a HuggingFace token to access the FineWeb-edu dataset. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).


## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```


## Submit the job

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/fineweb_dedup
```

Submit the job. Use `--env` to forward your Hugging Face token to authenticate with Hugging Face.

```bash
anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN
```

Results will be written to `/mnt/shared_storage/fineweb_processed/{timestamp}/` in the Parquet format.


## Understanding the example

- The pipeline performs three main stages on the text data:
    - **Text Cleaning**: Remove HTML tags, links, emails, normalize whitespace, and fix unicode characters.
    - **Text Filtering**: Filter documents by text length, alphanumeric ratio, special character ratio, and character repetition patterns.
    - **Deduplication**: Ray-based MinHash deduplication to remove near-duplicate documents.
- The entire pipeline is orchestrated by [Ray Data](https://docs.ray.io/en/latest/data/data.html), which handles distributed execution, fault tolerance, and resource management across your cluster.
- This example uses [Data-Juicer's Ray integration](https://github.com/modelscope/data-juicer) to run data processing operators at scale.
- Some notes on configuration:
    - This example passes `concurrency=20` into `ray.data.read_parquet` to control the rate of reading from Hugging Face and avoid bandwidth quota errors.
    - This example calls `repartition(target_num_rows_per_block=500)` after the `read_parquet` call to create smaller blocks for better parallelism and memory efficiency during processing.
