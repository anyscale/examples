# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import os
import time

from helper import parquet_to_webdataset_ray

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.deduplication.removal import ImageDuplicatesRemovalStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter


@dataclass
class Config:
    """Configuration loaded from environment variables."""
    input_parquet: str
    input_wds_dataset_dir: str
    output_dataset_dir: str
    embeddings_dir: str
    removal_parquets_dir: str
    model_dir: str
    entries_per_tar: int
    max_entries: int | None
    tar_files_per_partition: int
    batch_size: int
    embedding_batch_size: int

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        max_entries_str = os.environ.get("MAX_ENTRIES")
        
        return cls(
            input_parquet=os.environ["INPUT_PARQUET"],
            input_wds_dataset_dir=os.environ["INPUT_WDS_DIR"],
            output_dataset_dir=os.environ["OUTPUT_DIR"],
            embeddings_dir=os.environ["EMBEDDINGS_DIR"],
            removal_parquets_dir=os.environ["REMOVAL_DIR"],
            model_dir=os.environ.get("MODEL_DIR", "/home/ray/model_weights"),
            entries_per_tar=int(os.environ.get("ENTRIES_PER_TAR", "1000")),
            max_entries=int(max_entries_str) if max_entries_str else None,
            tar_files_per_partition=int(os.environ.get("TAR_FILES_PER_PARTITION", "1")),
            batch_size=int(os.environ.get("BATCH_SIZE", "100")),
            embedding_batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")),
        )


def create_image_embedding_pipeline(config: Config) -> Pipeline:
    """Create pipeline: read images -> generate CLIP embeddings -> save to parquet."""
    pipeline = Pipeline(name="image_embedding")

    pipeline.add_stage(FilePartitioningStage(
        file_paths=config.input_wds_dataset_dir,
        files_per_partition=config.tar_files_per_partition,
        file_extensions=[".tar"],
    ))

    pipeline.add_stage(ImageReaderStage(batch_size=config.batch_size))

    pipeline.add_stage(ImageEmbeddingStage(
        model_dir=config.model_dir,
        model_inference_batch_size=config.embedding_batch_size,
        remove_image_data=True,
    ))

    pipeline.add_stage(ConvertImageBatchToDocumentBatchStage(fields=["image_id", "embedding"]))

    pipeline.add_stage(ParquetWriter(path=config.embeddings_dir))

    return pipeline


def create_embedding_deduplication_workflow(config: Config) -> Pipeline:
    """Create semantic deduplication workflow using K-means + DBSCAN."""
    return SemanticDeduplicationWorkflow(
        input_path=config.embeddings_dir,
        output_path=config.removal_parquets_dir,
        id_field="image_id",
        embedding_field="embedding",
        n_clusters=100,
        eps=0.01,
    )


def create_image_deduplication_pipeline(config: Config) -> Pipeline:
    """Create pipeline: read images -> filter duplicates -> write deduplicated dataset."""
    pipeline = Pipeline(name="image_deduplication")

    pipeline.add_stage(FilePartitioningStage(
        file_paths=config.input_wds_dataset_dir,
        files_per_partition=config.tar_files_per_partition,
        file_extensions=[".tar"],
    ))

    pipeline.add_stage(ImageReaderStage(batch_size=config.batch_size))

    pipeline.add_stage(ImageDuplicatesRemovalStage(
        removal_parquets_dir=config.removal_parquets_dir + "/duplicates",
        duplicate_id_field="id",
    ))

    pipeline.add_stage(ImageWriterStage(
        output_dir=config.output_dataset_dir,
        remove_image_data=True,
    ))

    return pipeline


def main(config: Config) -> None:
    """Main execution function for image semantic deduplication pipeline."""
    ray_client = RayClient()
    ray_client.start()

    # Step 1: Download images and create WebDataset tar files
    os.makedirs(config.input_wds_dataset_dir, exist_ok=True)
    stats = parquet_to_webdataset_ray(
        hf_dataset_path=config.input_parquet,
        output_dir=config.input_wds_dataset_dir,
        entries_per_tar=config.entries_per_tar,
        max_entries=config.max_entries,
    )
    print(stats)

    # Use executors that avoid scheduling on CPU-only head node
    streaming_executor = RayDataExecutor(ignore_head_node=True)
    actor_executor = RayActorPoolExecutor(ignore_head_node=True)

    # Step 2: Generate CLIP embeddings
    pipeline = create_image_embedding_pipeline(config)
    pipeline.run(executor=streaming_executor)

    # Step 3: Find semantic duplicates using K-means + DBSCAN
    workflow = create_embedding_deduplication_workflow(config)
    workflow.run(kmeans_executor=actor_executor, pairwise_executor=actor_executor)

    # Step 4: Write deduplicated dataset
    pipeline = create_image_deduplication_pipeline(config)
    pipeline.run(executor=streaming_executor)

    ray_client.stop()


if __name__ == "__main__":
    config = Config.from_env()
    main(config)