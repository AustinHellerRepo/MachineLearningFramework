from __future__ import annotations
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import torch
import PIL.Image
from typing import List, Tuple, Dict
from austin_heller_repo.common import is_directory_empty, delete_directory_contents
from src.austin_heller_repo.machine_learning.dataset.dataset import DatasetSourceEnum, Dataset, KaggleDatasetEnum
from src.austin_heller_repo.machine_learning.framework import TensorCache, CharacterSetEnum, get_index_from_character, get_float_tensor_from_image, TensorCacheCategorySet


class Gpiosenka100BirdSpeciesSharpenKaggleDataset(Dataset):

	def __init__(self):
		super().__init__()
		pass

	@classmethod
	def get_dataset(cls) -> KaggleDatasetEnum:
		return KaggleDatasetEnum.Gpiosenka_100BirdSpecies_Sharpen

	@classmethod
	def get_dataset_source(cls) -> DatasetSourceEnum:
		return DatasetSourceEnum.Kaggle

	def get_tensor_cache_category_set_name(self) -> str:
		return "gpiosenka/100-bird-species/sharpen"

	def get_tensor_cache_category_set_input_tensor_size(self) -> Tuple[int, ...]:
		return (3, 223, 223)

	def get_tensor_cache_category_set_output_tensor_sizes(self) -> List[Tuple[int, ...]]:
		return [
			(3, 223, 223)
		]

	def download_to_directory(self, *, directory_path: str, is_forced: bool):

		download_directory_path = os.path.join(directory_path, "gpiosenka", "100-bird-species")

		if os.path.exists(download_directory_path) and \
			not is_directory_empty(
				directory_path=download_directory_path
			):

			if is_forced:
				delete_directory_contents(
					directory_path=download_directory_path
				)
				is_download_required = True
			else:
				is_download_required = False
		else:
			is_download_required = True

		if is_download_required:
			api_client = KaggleApi()
			api_client.authenticate()
			api_client.dataset_download_files(
				dataset="gpiosenka/100-bird-species",
				path=download_directory_path,
				unzip=True
			)

	def convert_to_tensor(self, *, download_directory_path: str, tensor_cache_directory_path: str):

		directory_path = os.path.join(download_directory_path, "gpiosenka", "100-bird-species")

		tensor_cache = TensorCache(
			cache_directory_path=tensor_cache_directory_path
		)

		tensor_cache.clear()

		input_tensor_size = self.get_tensor_cache_category_set_input_tensor_size()
		output_tensor_sizes = self.get_tensor_cache_category_set_output_tensor_sizes()
		if len(output_tensor_sizes) != 1:
			raise Exception(f"Unexpected number of output tensor sizes: {len(output_tensor_sizes)}.")

		tensor_cache_category_set = tensor_cache.create_tensor_cache_category_set(
			name=self.get_tensor_cache_category_set_name(),
			input_tensor_size=input_tensor_size,
			output_tensor_size=output_tensor_sizes[0]
		)

		for subdirectory_name in ["train", "test"]:
			subdirectory_path = os.path.join(directory_path, subdirectory_name)
			for bird_directory_name in os.listdir(subdirectory_path):
				bird_directory_path = os.path.join(subdirectory_path, bird_directory_name)
				for bird_image_file_name in os.listdir(bird_directory_path):
					bird_image_file_path = os.path.join(bird_directory_path, bird_image_file_name)
					image = PIL.Image.open(bird_image_file_path)

					expected_image = image.crop((0, 0, 223, 223))
					image_module_output = get_float_tensor_from_image(
						image=expected_image
					).view((3, 223, 223))

					blurred_image = expected_image.resize((112, 112)).resize((223, 223))
					image_module_input = get_float_tensor_from_image(
						image=blurred_image
					).view((3, 223, 223))

					tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
						name=subdirectory_name,
						output_tensor=image_module_output
					)

					tensor_cache_category_subset.create_tensor_cache_element_input(
						input_tensor=image_module_input
					)
