from __future__ import annotations
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import torch
import PIL.Image
from typing import List, Tuple, Dict
from sklearn import datasets
from austin_heller_repo.common import is_directory_empty, delete_directory_contents
from src.austin_heller_repo.machine_learning.dataset.dataset import DatasetSourceEnum, Dataset, SklearnDatasetEnum
from src.austin_heller_repo.machine_learning.framework import TensorCache, TensorCacheCategorySet, TensorCacheCategorySubset


class IrisSklearnDataset(Dataset):

	def __init__(self):
		super().__init__()
		pass

	@classmethod
	def get_dataset(cls) -> SklearnDatasetEnum:
		return SklearnDatasetEnum.Iris

	@classmethod
	def get_dataset_source(cls) -> DatasetSourceEnum:
		return DatasetSourceEnum.Sklearn

	def get_tensor_cache_category_set_name(self) -> str:
		return "iris"

	def get_tensor_cache_category_set_input_tensor_size(self) -> Tuple[int, ...]:
		return (4,)

	def get_tensor_cache_category_set_output_tensor_sizes(self) -> List[Tuple[int, ...]]:
		return [(1,)]

	def download_to_directory(self, *, directory_path: str, is_forced: bool):

		download_directory_path = os.path.join(directory_path, "iris")

		is_download_required = Dataset.is_download_required(
			download_directory_path=download_directory_path,
			is_forced=is_forced
		)

		if is_download_required:
			download_file_path = os.path.join(download_directory_path, "iris.txt")
			dataset = datasets.load_iris()
			with open(download_file_path, "w") as file_handle:
				for index, (features, target) in enumerate(zip(dataset.data, dataset.target)):
					if index != 0:
						file_handle.write("\n")
					file_handle.write(f"{target} {features[0]} {features[1]} {features[2]} {features[3]}")

	def convert_to_tensor(self, *, download_directory_path: str, tensor_cache_directory_path: str):

		file_path = os.path.join(download_directory_path, "iris", "iris.txt")

		tensor_cache = TensorCache(
			cache_directory_path=tensor_cache_directory_path
		)

		input_tensor_size = self.get_tensor_cache_category_set_input_tensor_size()
		output_tensor_sizes = self.get_tensor_cache_category_set_output_tensor_sizes()
		if len(output_tensor_sizes) != 1:
			raise Exception(f"Unexpected number of output tensor sizes: {len(output_tensor_sizes)}.")

		tensor_cache_category_set = tensor_cache.create_tensor_cache_category_set(
			name=self.get_tensor_cache_category_set_name(),
			input_tensor_size=input_tensor_size,
			output_tensor_size=output_tensor_sizes[0]
		)

		tensor_cache_category_subset_per_target = {}  # type: Dict[int, TensorCacheCategorySubset]

		with open(file_path, "r") as file_handle:
			for line in file_handle:
				target_string, feature_0_string, feature_1_string, feature_2_string, feature_3_string = (line.strip()).split(" ")
				target = int(target_string)
				feature_0, feature_1, feature_2, feature_3 = float(feature_0_string), float(feature_1_string), float(feature_2_string), float(feature_3_string)

				if target not in tensor_cache_category_subset_per_target:
					tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
						name=f"target_{target}",
						output_tensor=torch.LongTensor([target])
					)
					tensor_cache_category_subset_per_target[target] = tensor_cache_category_subset
				else:
					tensor_cache_category_subset = tensor_cache_category_subset_per_target[target]

				tensor_cache_category_subset.create_tensor_cache_element_input(
					input_tensor=torch.FloatTensor([
						feature_0,
						feature_1,
						feature_2,
						feature_3
					])
				)
