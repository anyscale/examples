"""Fuzzy dedup at scale implemented upon Ray. Pretty mush a fork of the code in DataJuicer."""

# uv run ray_minhash_lsh_datajuicer.py --flavor=datajuicer --documents_path=gs://anyscale-example-datasets/HuggingFaceFW/fineweb-edu/data/ --text_column=text --threshold=0.7 --ngram_size=5 --output=/mnt/cluster_storage/

import argparse
import hashlib
import logging
import os
import re
import string
import struct
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Annotated, cast
import numpy as np
import pyarrow as pa
import ray
from google.cloud import storage  # type: ignore
from numpy import typing as nptype
from pydantic import Field, PositiveInt
from scipy import integrate  # pyright: ignore[reportMissingTypeStubs]

BATCH_SIZE = 1000
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)

LOG_DEBUG = False

logger = logging.getLogger(__name__)


@dataclass
class WorkerParameters:
  """Contains the parameters needed by the tasks/actors of the
  dedup algorithm. This class should stay small as it is going to
  be transfered a lot to many tasks/actors"""

  num_workers: int = 5
  """Num KeyShardingActor. Increasing it will decrease the memory requirement
  per worker and fasten the processing of the dataset - it will however slightly
  increase the time it takes to calculate the global MinHashLSH (as a worker
  need to communicate with all its fellow workers).
  """

  num_minhash_tasks_per_worker: int = 10
  """Num min_hash_function tasks launched per KeyShardingActor to calculate
  ngrams and minhash.
  """

  num_similarity_tasks_per_worker: int = 10
  """Num similarity_function tasks launched per KeyShardingActor to calculate
  similarities.
  """

  threshold: float = 0.7
  """Threshold for the Jaccard similarity."""

  num_perm: int = 128
  """Permutation count for MinHash."""

  ngram_size: int = 5
  """Size of the char ngrams."""

  seed: int = 42
  """Seed of random generators."""

  def __post_init__(self):
    if self.num_workers < 1:
      raise ValueError('num_workers must be strictly positive')

    if self.threshold < 0.0 or self.threshold > 1.0:
      raise ValueError('threshold must be included in [0.0, 1.0]')

    if self.num_perm < 1:
      raise ValueError('num_perm must be strictly positive')

    if self.ngram_size < 1:
      raise ValueError('ngram_size must be strictly positive')


@dataclass
class DedupParameters:
  """Contains all the parameters of a dedup pipeline"""

  documents_path: str
  """Input path (could be a parquet file or a glob that points to
  multiple parquet files).
  """

  output_path: str
  """Output path. Will be used as a directory to write parquet files."""

  id_column: str
  """Name of the id column."""

  text_column: str
  """Name of the text column."""

  limit: int | None
  """Max number of documents to process."""

  dump_pairs_path: str | None
  """If not None, will dump all pairs returned by the
  MinHashLSH object (so before similarity calculation)
  in csv format. Useful for analysis.
  """

  display_forest_distribution: bool = False
  """If true, will display the forest cluster size
  distribution.
  """

  display_cluster_samples: int | None = None
  """If not None, will display some cluster samples."""

  work_dir: str = (
    os.path.join(os.environ['ANYSCALE_ARTIFACT_STORAGE'], f'rkn/dedup/{np.random.randint(1000000, 9999999)}/')
  )
  """Used by the datajuicer flavor to dump temporary data."""

  worker_parameters: WorkerParameters = field(default_factory=WorkerParameters)

  def __post_init__(self):
    if len(self.documents_path) == 0:
      raise ValueError('documents_path can not be empty')


class IdGenerator:
  def __init__(self, start_id: int = 0):
    self.next_id = start_id

  def get_next_id(self, count: int) -> tuple[int, int]:
    current_id = self.next_id
    self.next_id += count
    return (current_id, self.next_id)


class EdgeBuffer:
  def __init__(self):
    self.edge_dict = {}

  def clear(self):
    self.edge_dict = {}

  def set_edges(self, edge_dict):  # type: ignore
    self.edge_dict = edge_dict  # type: ignore

  def get_edges(self, key):  # type: ignore # noqa: ANN201
    return self.edge_dict.pop(key, [])  # type: ignore


class BTSUnionFind:
  """A distributed implementation of Union-Find with load balancing.

  The original paper on BTS Union-Find is available at:
  https://ieeexplore.ieee.org/document/10598116.

  TODO: export that in a file, unit test it.
  """

  def __init__(
    self,
    union_threshold: int,
    parallel_num: int,
    parallel_id: int,
    remote_edge_buffers: object,
    max_pending_edge_buffer_task: int,
    num_edge_buffer_task_returns: int,
  ):
    """Initialization method.

    :param union_threshold: threshold for minhash values group to
        perform union-find algorithm.
    :param parallel_num: number of parallel workers for
        union-find algorithm.
    :param parallel_id: unique id of this union find.
    :param remote_edge_buffers: ray-remote view of all the
        EdgeBuffer actors.
    :param max_pending_edge_buffer_task: max number of pending edge buffer
        ray tasks.
    :param num_edge_buffer_task_returns: number of edge buffer tasks for
        `ray.wait` to return.
    """
    self._union_threshold = union_threshold
    self._parallel_num = parallel_num
    self._parallel_id = parallel_id
    self._hash_table: dict[int, list[int]] = {}
    self._parent: dict[int, int] = {}
    self._old_parent: dict[int, int] = {}
    self._remote_edge_buffers = remote_edge_buffers
    self._edge_buffer = []
    self._edge_list_dict = {}
    self._max_pending_edge_buffer_task = max_pending_edge_buffer_task
    self._num_edge_buffer_task_returns = num_edge_buffer_task_returns

  def add_key_value_pairs(self, pairs: list[tuple[int, int]]) -> None:
    for key, value in pairs:
      if key not in self._hash_table:
        self._hash_table[key] = []
      self._hash_table[key].append(value)
      if len(self._hash_table[key]) > self._union_threshold:
        self._hash_table[key] = [self._union_list(self._hash_table[key])]

  def flush_key_value_pairs(self):
    for value in self._hash_table.values():
      if len(value) > 1:
        self._union_list(value)
    del self._hash_table

  def balanced_union_find(self):  # noqa: ANN201
    for x, y in self._edge_buffer:  # type: ignore
      self.union(x, y)  # type: ignore
    self._edge_buffer = []
    result_refs = []
    for remote_edge_buffer in self._remote_edge_buffers:  # type: ignore
      if len(result_refs) > self._max_pending_edge_buffer_task:  # type: ignore
        ready_refs, result_refs = ray.wait(  # type: ignore
          result_refs, num_returns=self._num_edge_buffer_task_returns
        )
        edge_list = ray.get(ready_refs)  # type: ignore
        for edges in edge_list:
          for x, y in edges:
            self.union(x, y)
        del ready_refs
      result_refs.append(remote_edge_buffer.get_edges.remote(self._parallel_id))  # type: ignore
    edge_list = ray.get(result_refs)  # type: ignore
    for edges in edge_list:
      for x, y in edges:
        self.union(x, y)
    del edge_list, result_refs
    self.rebalancing()
    return self._old_parent != self._parent

  def distribute_edge(self, u, v):  # type: ignore
    hash_u = u // BATCH_SIZE % self._parallel_num  # type: ignore
    hash_v = v // BATCH_SIZE % self._parallel_num  # type: ignore
    if hash_u not in self._edge_list_dict:  # type: ignore
      self._edge_list_dict[hash_u] = []  # type: ignore
    self._edge_list_dict[hash_u].append((u, v))  # type: ignore
    if hash_u != hash_v:
      if hash_v not in self._edge_list_dict:  # type: ignore
        self._edge_list_dict[hash_v] = []  # type: ignore
      self._edge_list_dict[hash_v].append((u, v))  # type: ignore

  def set_edge_buffer(self):
    if self._parallel_id in self._edge_list_dict:  # type: ignore
      self._edge_buffer = self._edge_list_dict[self._parallel_id]  # type: ignore
      del self._edge_list_dict[self._parallel_id]  # type: ignore
    else:
      self._edge_buffer = []
    ray.get(
      self._remote_edge_buffers[self._parallel_id].set_edges.remote(  # type: ignore
        self._edge_list_dict  # type: ignore
      )
    )
    self._edge_list_dict = {}

  def edge_redistribution(self):
    self.flush_key_value_pairs()
    self.rebalancing()
    self._edge_list_dict = {}
    for u, v in self._parent.items():
      self.distribute_edge(u, v)  # type: ignore
    self._parent = {}
    self.set_edge_buffer()

  def communication(self):
    self._edge_list_dict = {}
    del_list = []
    for u, v in self._parent.items():
      hash_u = u // BATCH_SIZE % self._parallel_num
      if self._parent[u] != self._old_parent.get(u, u) or (
        hash_u != self._parallel_id and v not in self._parent
      ):
        self.distribute_edge(u, v)  # type: ignore
      if hash_u != self._parallel_id:
        del_list.append(u)  # type: ignore
    self._old_parent = self._parent.copy()
    for u in del_list:  # type: ignore
      del self._parent[u]
    self.set_edge_buffer()

  def find(self, x: int) -> int:
    if x not in self._parent:
      return x
    else:
      self._parent[x] = self.find(self._parent[x])
      return self._parent[x]

  def union(self, x: int, y: int):
    px = self.find(x)
    py = self.find(y)
    if px == py:
      return
    if px > py:
      px, py = py, px
    self._parent[py] = px

  def _union_list(self, x_list: list[int]) -> int:
    px_list = [self.find(x) for x in x_list]
    p = min(px_list)
    for px in px_list:
      if p != px:
        self._parent[px] = p
    return p

  def rebalancing(self):
    new_px_dict = {}
    for x in self._parent:
      hash_x = x // BATCH_SIZE % self._parallel_num
      px = self.find(x)
      key = (px, hash_x)
      if key not in new_px_dict:
        new_px_dict[key] = x
      else:
        new_px_dict[key] = min(new_px_dict[key], x)  # type: ignore
    px_set = set(px for px, _ in new_px_dict)  # type: ignore
    for px in px_set:  # type: ignore
      hash_px = px // BATCH_SIZE % self._parallel_num  # type: ignore
      key = (px, hash_px)  # type: ignore
      if key not in new_px_dict:
        new_px_dict[key] = px
      else:
        new_px_dict[key] = min(new_px_dict[key], px)  # type: ignore

    for x in self._parent:
      hash_x = x // BATCH_SIZE % self._parallel_num
      px = self.find(x)
      key = (px, hash_x)
      if x == new_px_dict[key]:
        continue
      self._parent[x] = new_px_dict[key]

  def squeeze(self):
    dup_keys = {
      x
      for x in self._parent
      if x // BATCH_SIZE % self._parallel_num == self._parallel_id
    }
    self._parent = dup_keys  # type: ignore
    self._old_parent = {}
    self._edge_buffer = []
    ray.get(self._remote_edge_buffers[self._parallel_id].clear.remote())  # type: ignore

  def dup_idx(self, queries):  # noqa: ANN201  # type: ignore
    return [idx for uid, idx in queries if uid in self._parent]  # type: ignore


def get_remote_classes() -> dict[str, object]:
  """Get remote versions of classes with Ray decorators applied at runtime."""
  # Apply ray.method decorator to get_next_id at runtime
  IdGenerator.get_next_id = ray.method(num_returns=2)(IdGenerator.get_next_id)  # type: ignore

  return {
    'IdGenerator': ray.remote(IdGenerator),
    'EdgeBuffer': ray.remote(scheduling_strategy='SPREAD')(EdgeBuffer),
    'BTSUnionFind': ray.remote(memory=1e10, scheduling_strategy='SPREAD')(BTSUnionFind),
  }


def optimal_param(
  threshold: float,
  num_perm: int,
  false_positive_weight: float = 0.5,
  false_negative_weight: float = 0.5,
) -> tuple[int, int]:
  """Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
  of probabilities of false positive and false negative, taken from
  datasketch.

  :param threshold: float. The threshold for similarity
  :param num_perm: int. The number of permutations
  :param false_positive_weight: float. The weight of false positive
  :param false_negative_weight: float. The weight of false negative
  :return: Tuple[int, int]. The optimal `b` and `r` parameters. The number of
      bands, and the number of rows per band respectively
  """

  def false_positive_probability(th: float, band: int, rows: int) -> float:
    """Source: `datasketch.lsh`"""

    def proba(s: float) -> float:
      return 1 - (1 - s ** float(rows)) ** float(band)

    a: float
    a, _ = integrate.quad(proba, 0.0, th)  # type: ignore
    return a  # type: ignore

  def false_negative_probability(th: float, band: int, rows: int) -> float:
    """Source: `datasketch.lsh`"""

    def proba(s: float) -> float:
      return 1 - (1 - (1 - s ** float(rows)) ** float(band))

    a, _ = integrate.quad(proba, th, 1.0)  # type: ignore
    return a  # type: ignore

  # object: minimize the weighted FP and FN ratio
  min_error = float('inf')
  opt = (0, 0)
  for b in range(1, num_perm + 1):
    max_r = int(num_perm / b)
    for r in range(1, max_r + 1):
      fp = false_positive_probability(threshold, b, r)
      fn = false_negative_probability(threshold, b, r)
      error = fp * false_positive_weight + fn * false_negative_weight
      if error < min_error:
        min_error = error
        opt = (b, r)
  return opt


def sha1_hash32(data: bytes) -> int:
  """Directly taken from datasketch package to avoid dependency.

  Parameters
  ----------
  data : bytes

  Returns
  -------
  int
  """
  return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


# Default prefix for columns dynamically added to dataset.
DEFAULT_PREFIX = '__rf__'


# Name of columns dynamically added to dataset
class HashKeys:
  uid = DEFAULT_PREFIX + 'uid'


class RayBTSMinhashDeduplicator:
  """A MinhashLSH deduplicator based on RAY."""

  # TODO: Set a more reasonable value
  EMPTY_HASH_VALUE = 'EMPTY'
  _batched_op = True

  # TODO this mimicks the datajuicer machinery, we should make it
  # a param instead
  def use_cuda(self) -> bool:
    return False

  def __init__(
    self,
    tokenization: str = 'space',
    window_size: PositiveInt = 5,
    lowercase: bool = True,
    ignore_pattern: str | None = None,
    num_permutations: PositiveInt = 256,
    jaccard_threshold: Annotated[float, Field(ge=0, le=1)] = 0.7,
    num_bands: PositiveInt | None = None,
    num_rows_per_band: PositiveInt | None = None,
    tokenizer_model: str | None = None,
    union_find_parallel_num: int | str = 'auto',
    union_threshold: int | None = 256,
    max_pending_edge_buffer_task: int | None = 20,
    num_edge_buffer_task_returns: int | None = 10,
    max_pending_filter_tasks: int | None = 20,
    num_filter_task_returns: int | None = 10,
    merge_batch_size: int | None = 1000,
    minhash_batch_size: int | str | None = 'auto',
    memory_per_sample: float | None = 0.1,  # MB per sample
    # TODO added
    work_dir: str = tempfile.gettempdir(),
    text_key: str = 'text',
    seed: int | None = None,
    # TODO only for debug!!! also worst name ever
    execute_remote_calls_locally: bool = False,
    *args,  # type: ignore
    **kwargs,  # type: ignore
  ):
    """Initialization method.

    :param tokenization: tokenization method for sample texts. It
        should be one of [space, punctuation, character,
        sentencepiece]. For English-like languages, we recommend
        to use 'space', for Chinese-like languages, we recommend
        to use 'character', and for multiple languages, we recommend
        to use 'sentencepiece'. If using 'sentencepiece', please
        provided the model path in the 'tokenizer_model' field.
    :param window_size: window size of shingling
    :param lowercase: whether to convert text to lower case first
    :param ignore_pattern: whether to ignore sub-strings with
        specific pattern when computing minhash
    :param num_permutations: number of permutations in minhash
        computing
    :param jaccard_threshold: the min jaccard similarity threshold
        in near-duplicate detection. When the jaccard similarity of
        two sample texts is >= this threshold, they are regarded as
        similar samples and this op will only keep one of them after
        deduplication
    :param num_bands: number of bands in LSH. Default it's None, and
        it will be determined by an optimal params computation
        algorithm by minimize the weighted sum of probs of False
        Positives and False Negatives
    :param num_rows_per_band: number of rows in each band in LSH.
        Default it's None, and it will be determined by an optimal
        params computation algorithm
    :param tokenizer_model: path for the sentencepiece model, used for
        sentencepiece tokenization.
    :param union_find_parallel_num: number of parallel workers for
        union-find algorithm. Default it's 'auto', and it will be
        determined by half of the number of CPUs.
    :param union_threshold: threshold for minhash values group to
        perform union-find algorithm. Default it's 256.
    :param max_pending_edge_buffer_task: max number of pending edge buffer
        ray tasks. Default it's 20.
    :param num_edge_buffer_task_returns: number of edge buffer tasks for
        `ray.wait` to return. Default it's 10.
    :param max_pending_filter_tasks: max number of pending filter ray
        tasks. Default it's 20.
    :param num_filter_task_returns: number of filter tasks for `ray.wait`
        to return. Default it's 10.
    :param merge_batch_size: batch size for BTS operations. Default
        it's 1000.
    :param minhash_batch_size: batch size for MinHash computation. If "auto",
        it will be set to default value on CPU(1024), or auto calculated per
        available GPU memory and memory_per_sample setting for GPU.
    :param memory_per_sample: estimated memory needed per sample in MB.
        Used to calculate batch size based on available GPU memory.
        Default is 0.1 MB per sample.
    """

    # super().__init__(*args, **kwargs)

    self.tokenization = tokenization
    self.window_size = window_size
    self.lowercase = lowercase
    self.memory_per_sample = memory_per_sample
    if minhash_batch_size == 'auto':
      if self.use_cuda():
        self.minhash_batch_size = 200_000
      else:
        self.minhash_batch_size = 1024
    else:
      self.minhash_batch_size = minhash_batch_size
    self.ignore_pattern: re.Pattern[str] | None = None
    if ignore_pattern:
      self.ignore_pattern = re.compile(ignore_pattern)
    if self.use_cuda() and self.tokenization != 'character':
      raise ValueError(
        'GPU MinHash computation is only supported for character tokenization'
      )

    # check parameters
    if self.ignore_pattern and self.tokenization == 'punctuation':
      logger.warning(
        'Be careful that tokenization with punctuations '
        "won't work if the ignore pattern includes "
        'punctuations.'
      )
    # self.punctuation_pattern = re.compile(r'\p{P}')
    self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')

    if self.tokenization == 'sentencepiece':
      if tokenizer_model is None:
        raise ValueError(
          "To use 'sentencepiece' tokenization, 'tokenizer_model' is required."
        )
      # self.tokenizer = prepare_sentencepiece_model(tokenizer_model)
      raise NotImplementedError("'sentencepiece' tokenization is not supported")
    else:
      self.tokenizer = None

    tokenization_func: Callable[[str], set[bytes]] | None = None

    if self.tokenization == 'character':

      def char_tokenization_func(text: str) -> set[bytes]:
        return {
          str.encode(text[i : i + self.window_size])
          for i in range(len(text) - self.window_size + 1)
        }

      tokenization_func = char_tokenization_func

    elif self.tokenization == 'punctuation':

      def punctuation_tokenization_func(text: str) -> set[bytes]:
        tokens = self.punctuation_pattern.split(text)
        return {
          str.encode(' '.join(tokens[i : i + self.window_size]))
          for i in range(len(tokens) - self.window_size + 1)
        }

      tokenization_func = punctuation_tokenization_func

    elif self.tokenization == 'space':
      raise NotImplementedError("'space' tokenizer is not implemented")

      # def space_tokenization_func(text: str) -> set[bytes]:
      #   tokens = split_on_whitespace(text)
      #   return {
      #     str.encode(' '.join(tokens[i : i + self.window_size]))
      #     for i in range(len(tokens) - self.window_size + 1)
      #   }

      # tokenization_func = space_tokenization_func

    elif self.tokenization == 'sentencepiece':
      raise NotImplementedError("'sentencepiece' tokenizer is not implemented")

    #   def sentencepiece_tokenization_func(text: str) -> set[bytes]:
    #     tokens = self.tokenizer.encode(text, out_type=str)
    #     return {
    #       str.encode(''.join(tokens[i : i + self.window_size]))
    #       for i in range(len(tokens) - self.window_size + 1)
    #     }

    #   tokenization_func = sentencepiece_tokenization_func

    else:
      raise NotImplementedError(
        f'Unimplemented tokenization method [{self.tokenization}]'
      )
    self.tokenization_func = tokenization_func

    # about deduplication
    self.num_permutation = num_permutations
    self.jaccard_threshold = jaccard_threshold
    self.num_bands = num_bands
    self.num_rows_per_band = num_rows_per_band

    # initialize deduplication parameters
    # check number of bands and rows
    if self.num_bands is None or self.num_rows_per_band is None:
      self.num_bands, self.num_rows_per_band = optimal_param(
        self.jaccard_threshold,
        self.num_permutation,
      )

    # compute hash ranges and create hash tables
    self.hash_ranges: list[tuple[int, int]] = [
      (i * self.num_rows_per_band, (i + 1) * self.num_rows_per_band)
      for i in range(self.num_bands)
    ]

    # generate permutations
    gen = np.random.RandomState(seed=seed)
    self.perm_a: nptype.NDArray[np.uint64]
    self.perm_b: nptype.NDArray[np.uint64]
    self.perm_a, self.perm_b = np.array(
      [
        (
          gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),  # type: ignore
          gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),  # type: ignore
        )
        for _ in range(self.num_permutation)
      ],
      dtype=np.uint64,
    ).T

    if union_find_parallel_num == 'auto':
      union_find_parallel_num = max(
        1,
        int(ray.cluster_resources().get('CPU') / 2),  # type: ignore
      )
    else:
      union_find_parallel_num = int(union_find_parallel_num)

    self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
    self.num_edge_buffer_task_returns = num_edge_buffer_task_returns
    self.max_pending_filter_tasks = max_pending_filter_tasks
    self.num_filter_task_returns = num_filter_task_returns
    if merge_batch_size is None:
      self.merge_batch_size = union_find_parallel_num
    else:
      self.merge_batch_size = min(merge_batch_size, union_find_parallel_num)

    logger.info(f'union_find_parallel_num = {union_find_parallel_num}')
    self.union_find_parallel_num = union_find_parallel_num
    self.union_threshold = union_threshold

    # Get remote classes only when needed
    remote_classes = get_remote_classes()
    self.remote_edge_buffers = [  # type: ignore
      remote_classes['EdgeBuffer'].remote()  # type: ignore
      for _ in range(self.union_find_parallel_num)
    ]
    self.union_find_list = [  # type: ignore
      remote_classes['BTSUnionFind'].remote(  # type: ignore
        self.union_threshold,
        self.union_find_parallel_num,
        i,
        self.remote_edge_buffers,  # type: ignore
        self.max_pending_edge_buffer_task,
        self.num_edge_buffer_task_returns,
      )
      for i in range(self.union_find_parallel_num)
    ]

    empty_hash_value: nptype.NDArray[np.uint32] = np.full(
      (self.num_rows_per_band,), MAX_HASH, dtype=np.uint32
    )
    self.empty_hash_value: bytes = (
      b'\x00\x00\x00\x00' + empty_hash_value.tobytes()
    )
    self.empty_hash_table_id: int = int(MAX_HASH % self.union_find_parallel_num)

    self.work_dir = work_dir
    self.text_key = text_key

    self.execute_remote_calls_locally = execute_remote_calls_locally

  def band_minhash(self, minhash_list, uid_list):  # type: ignore
    """Logic for creating and pusing LSH bands to the union find list"""
    pairs = {}
    minhash_list = minhash_list.to_numpy(zero_copy_only=False)  # type: ignore
    for minhash, uid in zip(minhash_list, uid_list):  # type: ignore
      for i, (start, end) in enumerate(self.hash_ranges):
        hash_value = i.to_bytes(4, 'big') + minhash[start:end].tobytes()  # type: ignore
        hash_table_id = minhash[start] % self.union_find_parallel_num  # type: ignore
        if hash_table_id not in pairs:
          pairs[hash_table_id] = []
        pairs[hash_table_id].append((hash_value, uid))  # type: ignore
    result_refs = []
    for i, p in pairs.items():  # type: ignore
      if len(result_refs) > self.max_pending_filter_tasks:  # type: ignore
        ready_refs, result_refs = ray.wait(  # type: ignore
          result_refs, num_returns=self.num_filter_task_returns
        )
        ray.get(ready_refs)  # type: ignore
      result_refs.append(self.union_find_list[i].add_key_value_pairs.remote(p))  # type: ignore
    ray.get(result_refs)  # type: ignore

  def calc_minhash(
    self,
    text_list: pa.Array,  # type: ignore
    uid_list: Iterable[int],
  ) -> pa.Table:  # type: ignore
    """Logic for computing minhash values for each text in the input table"""
    pairs: dict[int, list[tuple[bytes, int]]] = {}
    for text, uid in zip(text_list, uid_list):  # type: ignore
      text = cast(pa.StringScalar, text)
      str_text: str = text.as_py()
      if self.lowercase:
        str_text = str_text.lower()
      if self.ignore_pattern:
        str_text = self.ignore_pattern.sub('', str_text)

      tokens: set[bytes] = self.tokenization_func(str_text)
      if len(tokens) > 0:
        hv: nptype.NDArray[np.uint64] = np.array(
          [sha1_hash32(token) for token in tokens], dtype=np.uint64
        )
        phv: nptype.NDArray[np.uint32] = (
          (hv[:, None] * self.perm_a[None, :] + self.perm_b) % MERSENNE_PRIME
        ).astype(np.uint32)
        hash_values: nptype.NDArray[np.uint32] = phv.min(axis=0)
        for i, (start, end) in enumerate(self.hash_ranges):
          hash_value: bytes = (
            i.to_bytes(4, 'big') + hash_values[start:end].tobytes()
          )
          hash_table_id: int = hash_values[start] % self.union_find_parallel_num
          if hash_table_id not in pairs:
            pairs[hash_table_id] = []
          pairs[hash_table_id].append((hash_value, uid))
      else:
        if self.empty_hash_table_id not in pairs:
          pairs[self.empty_hash_table_id] = []
        pairs[self.empty_hash_table_id].append((self.empty_hash_value, uid))

    result_refs = []
    for i, p in pairs.items():
      if len(result_refs) > self.max_pending_filter_tasks:  # type: ignore
        ready_refs, result_refs = ray.wait(  # type: ignore
          result_refs, num_returns=self.num_filter_task_returns
        )
        ray.get(ready_refs)  # type: ignore
      result_refs.append(self.union_find_list[i].add_key_value_pairs.remote(p))  # type: ignore
    ray.get(result_refs)  # type: ignore

  def merge_op_batch(self, object_refs):  # type: ignore # noqa: ANN201
    results = []
    while object_refs:
      ready_refs, object_refs = ray.wait(  # type: ignore
        object_refs, num_returns=min(self.merge_batch_size, len(object_refs))
      )
      results.extend(ray.get(ready_refs))  # type: ignore
    return results  # type: ignore

  def merge(self):
    self.merge_op_batch(  # type: ignore
      [
        union_find.edge_redistribution.remote()  # type: ignore
        for union_find in self.union_find_list  # type: ignore
      ]
    )
    while any(
      self.merge_op_batch(  # type: ignore
        [
          union_find.balanced_union_find.remote()  # type: ignore
          for union_find in self.union_find_list  # type: ignore
        ]
      )
    ):
      self.merge_op_batch(  # type: ignore
        [
          union_find.communication.remote()  # type: ignore
          for union_find in self.union_find_list  # type: ignore
        ]
      )
    self.merge_op_batch(  # type: ignore
      [union_find.squeeze.remote() for union_find in self.union_find_list]  # type: ignore
    )

  def filter_with_union_find(self, samples: pa.Table) -> pa.Table:
    query_dict = {}
    for idx, uid in enumerate(samples[HashKeys.uid]):
      uid = uid.as_py()
      hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
      if hash_id not in query_dict:
        query_dict[hash_id] = []
      query_dict[hash_id].append((uid, idx))  # type: ignore
    mask = np.ones(len(samples), dtype=np.bool_)
    result_refs = []
    for hash_id, query in query_dict.items():  # type: ignore
      if len(result_refs) > self.max_pending_filter_tasks:  # type: ignore
        ready_refs, result_refs = ray.wait(  # type: ignore
          result_refs, num_returns=self.num_filter_task_returns
        )
        results = ray.get(ready_refs)  # type: ignore
        for result in results:
          mask[result] = False
        del ready_refs
      result_refs.append(self.union_find_list[hash_id].dup_idx.remote(query))  # type: ignore
    results = ray.get(result_refs)  # type: ignore
    for result in results:
      mask[result] = False
    del query_dict, results
    columns_to_keep = [
      name for name in samples.column_names if name != HashKeys.uid
    ]
    return samples.select(columns_to_keep).filter(mask)

  def run(self, dataset, **kwargs):  # type: ignore # noqa: ANN201
    # Ignore additional parameters like exporter, tracer, etc.
    start_time = time.time()
    # Get remote IdGenerator only when needed
    remote_classes = get_remote_classes()
    id_generator = remote_classes['IdGenerator'].remote()  # type: ignore

    def minhash_with_uid(table: pa.Table) -> pa.Table:
      num_rows = len(table)
      min_id, max_id = ray.get(id_generator.get_next_id.remote(num_rows))  # type: ignore
      uid_list = range(min_id, max_id)
      self.calc_minhash(table[self.text_key], uid_list)  # type: ignore
      new_table = table.append_column(HashKeys.uid, pa.array(list(uid_list)))
      return new_table

    tmp_dir = os.path.join(
      self.work_dir,
      '.tmp',
      ray.get_runtime_context().get_job_id(),  # type: ignore
    )
    if self.use_cuda():
      raise ValueError('use_cuda not supported yet for dedup')
    else:
      logger.info('Using CPU for MinHash computation')
      dataset.map_batches(  # type: ignore
        minhash_with_uid,
        batch_format='pyarrow',
        zero_copy_batch=True,
        memory=2e10
      ).write_parquet(
        tmp_dir,
        try_create_dir=False,
        ray_remote_args={"memory": 1e10}
      )
      # TODO: del dataset too ?
    end_time = time.time()
    logger.info(f'MinHash time = {end_time - start_time}')
    new_dataset = ray.data.read_parquet(tmp_dir)  # type: ignore
    start_time = time.time()
    self.merge()
    end_time = time.time()
    logger.info(f'merge time = {end_time - start_time}')
    start_time = time.time()
    result = new_dataset.map_batches(  # type: ignore
      self.filter_with_union_find,
      batch_format='pyarrow',
      zero_copy_batch=True,
    )
    end_time = time.time()
    logger.info(f'filter time = {end_time - start_time}')
    return result


# TODO: use olympus/storage/file when python version has been updated on the Ray cluster (we
# need @override).
def _bucket_and_blob_path_from_uri(
  uri: str, project: str
) -> tuple[storage.Bucket, str]:
  """Extract a bucket and a blob path from a uri. Use the provided project to build the storage.Client"""
  assert uri.startswith('gs://'), "GCS RPath URIs must start with 'gs://'"
  split = uri.removeprefix('gs://').split('/', 1)
  bucket_path = split[0]
  # We do support GcsRPath that points to the root of the bucket, in which case there will
  # be no blob_path.
  blob_path = ''
  if len(split) > 1:
    blob_path = split[1]
  if not bucket_path:
    raise ValueError('GCS RPath URIs must contain a bucket')
  client = storage.Client(project=project)
  bucket = client.bucket(bucket_path)  # type: ignore
  return (bucket, blob_path)


def execute(parameters: DedupParameters) -> None:
  start_time = time.time()

  # Process documents
  input_path: list[str] = []
  for document_path in parameters.documents_path.split(','):
    if document_path.find('*') != -1:
      logging.info(f'Input {document_path} has a star, will do a glob search')
      # Glob pattern.
      index_of_first_star = document_path.index('*')
      index_of_previous_slash = document_path.rfind('/', 0, index_of_first_star)
      if index_of_previous_slash == -1:
        raise ValueError('TODO')

      main = document_path[0 : index_of_previous_slash + 1]
      glob = '**/' + document_path[index_of_previous_slash + 1 :]

      bucket, blob_path = _bucket_and_blob_path_from_uri(main)
      # root_blob = bucket.blob(blob_name=blob_path)

      for subblob in bucket.list_blobs(prefix=blob_path, match_glob=glob):  # type: ignore
        input_path.append(f'gs://{bucket.name}/{subblob.name}')  # type: ignore
      logging.info(
        f'Found {len(input_path)} document(s) matching {document_path}'
      )
    else:
      logging.info(
        f'Input {document_path} is a file or a directory, will pass it directly to read_parquet(...)'
      )
      # Note that read_parquet does not like reading multiple directory - we would have to enumerate files inside instead.
      # TODO in the future.
      input_path.append(document_path)

  import gcsfs
  filesystem = gcsfs.GCSFileSystem(project="test-vertex")
  ds = ray.data.read_parquet(  # type: ignore
    input_path,
    filesystem=filesystem,
    ray_remote_args={"memory": 1e10}
  )

  # Filter data so we only have what matters to us: id and text.
  ds_less_columns = ds.select_columns(  # type: ignore
    [parameters.id_column, parameters.text_column]
  )

  if parameters.limit is not None:
    logger.info(
      f'Received a dataset, will only process {parameters.limit} documents'
    )
    # Note: we materialize the filtered ds, as if we don't two iterations
    # on ds_less_columns will yield different results, which will mess up
    # with the writing of the filtered data at the end.
    ds_less_columns = ds_less_columns.limit(parameters.limit).materialize()  # type: ignore
  else:
    logger.info('Received a dataset, will process all documents')

  dedup = RayBTSMinhashDeduplicator(
    tokenization='character',
    window_size=parameters.worker_parameters.ngram_size,
    lowercase=True,
    ignore_pattern=f'[{re.escape(string.punctuation)}]',
    num_permutations=parameters.worker_parameters.num_perm,
    jaccard_threshold=parameters.worker_parameters.threshold,
    work_dir=parameters.work_dir,
    text_key=parameters.text_column,
    seed=parameters.worker_parameters.seed,
  )
  output_ds = dedup.run(ds_less_columns)  # type: ignore

  if parameters.display_forest_distribution:
    # TODO
    pass

  if parameters.display_cluster_samples is not None:
    # TODO
    pass

  # TODO: logger.info(f'Filtered {filter_size} entries')

  end_time = time.time()
  logger.info(f'Processing completed in {end_time - start_time:.2f} seconds')

  # TODO stats = ray.get(coordinator.get_all_stats.remote())
  # TODO logger.info(f'Statistics: {stats}')

  # Save results if output file specified
  if parameters.output_path:
    # Note: writing parquet files in a directory does not work yet on gcs -
    # to be fixed when I have the time.
    output_ds.write_parquet(parameters.output_path, try_create_dir=False)  # type: ignore
    logger.info(f'Results saved to {parameters.output_path}')


def main() -> None:
  """Main function for Ray-based MinHash LSH."""

  # TODO: replace the argparse stuff with flags.
  parser = argparse.ArgumentParser(description='Ray-based MinHash LSH')

  parser.add_argument(
    '--flavor',
    type=str,
    required=True,
    help="Name of the algorithm to use. Possible values: ['presorting', 'datajuicer'].",
  )

  parser.add_argument(
    '--documents_path',
    type=str,
    required=True,
    help='Path to JSONL file or Parquet file/directory with documents (supports GCS gs:// paths)',
  )
  parser.add_argument(
    '--input_format',
    type=str,
    choices=['jsonl', 'parquet'],
    default='jsonl',
    help='Input format: jsonl or parquet',
  )
  parser.add_argument(
    '--text_column',
    type=str,
    default='text',
    help='Name of text column in parquet files',
  )
  parser.add_argument(
    '--id_column',
    type=str,
    default='id',
    help='Name of ID column in parquet files',
  )
  parser.add_argument(
    '--threshold', type=float, default=0.8, help='Jaccard similarity threshold'
  )
  parser.add_argument(
    '--num_workers',
    type=int,
    default=4,
    help='Number of Ray workers',
  )
  parser.add_argument(
    '--num_minhash_tasks_per_worker',
    type=int,
    default=10,
    help='Number of minhash tasks launched per worker',
  )
  parser.add_argument(
    '--num_perm',
    type=int,
    default=128,
    help='Number of hash functions for MinHash',
  )
  parser.add_argument(
    '--ngram_size', type=int, default=3, help='Size of n-grams'
  )
  parser.add_argument(
    '--limit',
    type=int,
    default=None,
    help='Limit number of documents to process',
  )
  parser.add_argument(
    '--output',
    type=str,
    default=None,
    help='Output path for filtered dataset (parquet root)',
  )
  parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Seed for MinHash',
  )
  parser.add_argument(
    '--display_cluster_samples',
    type=int,
    default=None,
    help='If set, will display a random set of clusters, with the chosen representative, the similar ids and their text',
  )
  parser.add_argument(
    '--display_forest_distribution',
    action='store_true',
    help='If true, will display the disjoint set cluster size distribution',
  )

  parser.add_argument(
    '--dump_pairs_path',
    type=str,
    default=None,
    help='If set, will dump list of similar pairs (before disjoint-set calculation)',
  )

  args = parser.parse_args()

  worker_parameters = WorkerParameters(
    num_workers=args.num_workers,
    num_minhash_tasks_per_worker=args.num_minhash_tasks_per_worker,
    threshold=args.threshold,
    num_perm=args.num_perm,
    ngram_size=args.ngram_size,
    seed=args.seed,
  )

  dedup_parameters = DedupParameters(
    documents_path=args.documents_path,
    output_path=args.output,
    id_column=args.id_column,
    text_column=args.text_column,
    limit=args.limit,
    dump_pairs_path=args.dump_pairs_path,
    display_forest_distribution=args.display_forest_distribution,
    display_cluster_samples=args.display_cluster_samples,
    worker_parameters=worker_parameters,
  )

  execute(dedup_parameters)


if __name__ == '__main__':
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
  )

  ray.init()  # type: ignore
  main()