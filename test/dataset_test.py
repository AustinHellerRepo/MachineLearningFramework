from __future__ import annotations
import unittest
import os
from datetime import datetime
from typing import List, Tuple, Dict
from src.austin_heller_repo.machine_learning.dataset.kaggle.nabeel965_handwritten_words_dataset import Nabeel965HandwrittenWordsDatasetKaggleDataset, Dataset, TensorCache
from austin_heller_repo.common import delete_directory_contents, get_all_files_in_directory


download_directory_path = "./cache/dataset"
tensor_directory_path = "./cache/tensor"


def get_default_datasets() -> List[Dataset]:
	return [
		Nabeel965HandwrittenWordsDatasetKaggleDataset()
	]


class DatasetTest(unittest.TestCase):

	def test_download_dataset_empty(self):

		for dataset in get_default_datasets():

			delete_directory_contents(
				directory_path=download_directory_path
			)
			delete_directory_contents(
				directory_path=tensor_directory_path
			)

			files = get_all_files_in_directory(
				directory_path=download_directory_path,
				include_subdirectories=True
			)

			self.assertEqual(0, len(files))

			dataset.download_to_directory(
				directory_path=download_directory_path,
				is_forced=False
			)

	def test_download_dataset_not_empty_not_forced(self):

		for dataset in get_default_datasets():

			delete_directory_contents(
				directory_path=download_directory_path
			)
			delete_directory_contents(
				directory_path=tensor_directory_path
			)

			# first
			dataset.download_to_directory(
				directory_path=download_directory_path,
				is_forced=False
			)

			modified_time_per_file_path = {}  # type: Dict[str, float]
			for file_path in get_all_files_in_directory(
				directory_path=download_directory_path,
				include_subdirectories=True
			):
				modified_time = os.path.getmtime(file_path)
				modified_time_per_file_path[file_path] = modified_time

			# second
			dataset.download_to_directory(
				directory_path=download_directory_path,
				is_forced=False
			)

			for file_path in get_all_files_in_directory(
					directory_path=download_directory_path,
					include_subdirectories=True
			):
				self.assertIn(file_path, modified_time_per_file_path)

				modified_time = os.path.getmtime(file_path)

				self.assertEqual(modified_time_per_file_path[file_path], modified_time)

	def test_download_dataset_not_empty_forced(self):

		for dataset in get_default_datasets():

			delete_directory_contents(
				directory_path=download_directory_path
			)
			delete_directory_contents(
				directory_path=tensor_directory_path
			)

			# first
			dataset.download_to_directory(
				directory_path=download_directory_path,
				is_forced=False
			)

			modified_time_per_file_path = {}  # type: Dict[str, float]
			for file_path in get_all_files_in_directory(
				directory_path=download_directory_path,
				include_subdirectories=True
			):
				modified_time = os.path.getmtime(file_path)
				modified_time_per_file_path[file_path] = modified_time

			# second
			dataset.download_to_directory(
				directory_path=download_directory_path,
				is_forced=True
			)

			for file_path in get_all_files_in_directory(
					directory_path=download_directory_path,
					include_subdirectories=True
			):
				self.assertIn(file_path, modified_time_per_file_path)

				modified_time = os.path.getmtime(file_path)

				self.assertNotEqual(modified_time_per_file_path[file_path], modified_time)

	def test_create_tensor_cache_cycle_runner(self):

		for dataset in get_default_datasets():

			delete_directory_contents(
				directory_path=download_directory_path
			)
			delete_directory_contents(
				directory_path=tensor_directory_path
			)

			dataset.download_to_directory(
				directory_path=download_directory_path,
				is_forced=False
			)

			dataset.convert_to_tensor(
				download_directory_path=download_directory_path,
				tensor_cache_directory_path=tensor_directory_path
			)

			tensor_cache_category_set = dataset.get_tensor_cache_category_set(
				tensor_cache_directory_path=tensor_directory_path
			)

			self.assertIsNotNone(tensor_cache_category_set)
