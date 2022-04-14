from __future__ import annotations
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import torch
import PIL.Image
from typing import List, Tuple, Dict
from austin_heller_repo.common import is_directory_empty, delete_directory_contents
from src.austin_heller_repo.machine_learning.dataset.kaggle.kaggle_dataset import KaggleDatasetEnum
from src.austin_heller_repo.machine_learning.dataset.dataset import DatasetSourceEnum, Dataset
from src.austin_heller_repo.machine_learning.framework import TensorCache, CharacterSetEnum, get_index_from_character, get_float_tensor_from_image, TensorCacheCategorySet


class Nabeel965HandwrittenWordsDatasetKaggleDataset(Dataset):

	def __init__(self):
		super().__init__()
		pass

	@classmethod
	def get_dataset(cls) -> KaggleDatasetEnum:
		return KaggleDatasetEnum.Nabeel965_HandwrittenWordsDataset

	@classmethod
	def get_dataset_source(cls) -> DatasetSourceEnum:
		return DatasetSourceEnum.Kaggle

	def get_tensor_cache_category_set_name(self) -> str:
		return "nabeel965/handwritten-words-dataset"

	def get_tensor_cache_category_set_input_tensor_size(self) -> Tuple[int, ...]:
		return (1, 80, 146)

	def get_tensor_cache_category_set_output_tensor_sizes(self) -> List[Tuple[int, ...]]:
		return [
			(4,),
			(5,),
			(7,),
			(8,)
		]

	def download_to_directory(self, *, directory_path: str, is_forced: bool):

		download_directory_path = os.path.join(directory_path, "nabeel965", "handwritten-words-dataset")

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
				dataset="nabeel965/handwritten-words-dataset",
				path=download_directory_path,
				unzip=True
			)

	def convert_to_tensor(self, *, download_directory_path: str, tensor_cache_directory_path: str):

		directory_path = os.path.join(download_directory_path, "nabeel965", "handwritten-words-dataset", "data")

		tensor_cache = TensorCache(
			cache_directory_path=tensor_cache_directory_path
		)

		tensor_cache.clear()

		input_tensor_size = self.get_tensor_cache_category_set_input_tensor_size()

		tensor_cache_category_set_per_output_tensor_size = {}  # type: Dict[Tuple[int, ...], TensorCacheCategorySet]

		word_directory_path_per_word = {}  # type: Dict[str, str]
		for subset_name in ["Capital", "small"]:
			for directory_name in os.listdir(os.path.join(directory_path, subset_name)):
				word_directory_path = os.path.join(directory_path, subset_name, directory_name, directory_name)
				word_directory_path_per_word[directory_name] = word_directory_path

		for word, word_directory_path in word_directory_path_per_word.items():
			output_tensor = torch.LongTensor([
				get_index_from_character(
					character=x,
					character_set=CharacterSetEnum.English
				) for x in word
			] + [0])

			output_tensor_size = tuple(output_tensor.shape)

			if output_tensor_size not in tensor_cache_category_set_per_output_tensor_size:
				tensor_cache_category_set = tensor_cache.create_tensor_cache_category_set(
					name=self.get_tensor_cache_category_set_name(),
					input_tensor_size=input_tensor_size,
					output_tensor_size=output_tensor_size
				)
				tensor_cache_category_set_per_output_tensor_size[output_tensor_size] = tensor_cache_category_set
			else:
				tensor_cache_category_set = tensor_cache_category_set_per_output_tensor_size[output_tensor_size]

			tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
				name=word,
				output_tensor=output_tensor
			)

			for file_name in os.listdir(word_directory_path):
				file_path = os.path.join(word_directory_path, file_name)
				image = PIL.Image.open(file_path)
				image_tensor = get_float_tensor_from_image(
					image=image
				)
				tensor_cache_category_subset.create_tensor_cache_element_input(
					input_tensor=image_tensor
				)
