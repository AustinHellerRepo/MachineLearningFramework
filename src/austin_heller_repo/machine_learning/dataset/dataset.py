from __future__ import annotations
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from austin_heller_repo.common import StringEnum
from src.austin_heller_repo.machine_learning.framework import TensorCache, TensorCacheCategorySet, TensorCacheCategorySubsetCycle, TensorCacheCategorySubsetSequence
from src.austin_heller_repo.machine_learning.service import AddTrainingDataAnnouncementServiceClientServerMessage


class DatasetSourceEnum(StringEnum):
	Kaggle = "kaggle"


class DatasetEnum(StringEnum):
	pass


class Dataset():

	def __init__(self):
		pass

	@classmethod
	@abstractmethod
	def get_dataset(cls) -> DatasetEnum:
		raise NotImplementedError()

	@classmethod
	@abstractmethod
	def get_dataset_source(cls) -> DatasetSourceEnum:
		raise NotImplementedError()

	@abstractmethod
	def get_tensor_cache_category_set_name(self) -> str:
		raise NotImplementedError()

	@abstractmethod
	def get_tensor_cache_category_set_input_tensor_size(self) -> Tuple[int, ...]:
		raise NotImplementedError()

	@abstractmethod
	def get_tensor_cache_category_set_output_tensor_sizes(self) -> List[Tuple[int, ...]]:
		raise NotImplementedError()

	@abstractmethod
	def download_to_directory(self, *, directory_path: str, is_forced: bool):
		raise NotImplementedError()

	@abstractmethod
	def convert_to_tensor(self, *, download_directory_path: str, tensor_cache_directory_path: str):
		raise NotImplementedError()

	def get_tensor_cache_category_sets(self, *, tensor_cache_directory_path: str) -> List[TensorCacheCategorySet]:
		tensor_cache = TensorCache(
			cache_directory_path=tensor_cache_directory_path
		)
		tensor_cache_category_sets = []  # type: List[TensorCacheCategorySet]
		for output_tensor_size in self.get_tensor_cache_category_set_output_tensor_sizes():
			tensor_cache_category_sets.extend(tensor_cache.get_tensor_cache_category_sets(
				name=self.get_tensor_cache_category_set_name(),
				input_tensor_size=self.get_tensor_cache_category_set_input_tensor_size(),
				output_tensor_size=output_tensor_size
			))

		return tensor_cache_category_sets

	def get_tensor_cache_category_subset_cycle(self, *, tensor_cache_directory_path: str) -> TensorCacheCategorySubsetCycle:

		tensor_cache_category_sets = self.get_tensor_cache_category_sets(
			tensor_cache_directory_path=tensor_cache_directory_path
		)

		return TensorCacheCategorySubsetCycle(
			tensor_cache_category_subset_sequences=[
				TensorCacheCategorySubsetSequence(
					tensor_cache_category_subsets=x.get_tensor_cache_category_subsets()
				)
				for x in tensor_cache_category_sets
			]
		)
