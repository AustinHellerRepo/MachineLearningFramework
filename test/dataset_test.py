from __future__ import annotations
import unittest
import os
from datetime import datetime
from typing import List, Tuple, Dict
import random
import time
from src.austin_heller_repo.machine_learning.dataset.kaggle.nabeel965_handwritten_words_dataset import Nabeel965HandwrittenWordsDatasetKaggleDataset, Dataset, TensorCache
from src.austin_heller_repo.machine_learning.dataset.sklearn.iris_dataset import IrisSklearnDataset
from src.austin_heller_repo.machine_learning.dataset.kaggle.gpiosenka_100_bird_species_sharpen_dataset import Gpiosenka100BirdSpeciesSharpenKaggleDataset
from austin_heller_repo.common import delete_directory_contents, get_all_files_in_directory


#download_directory_path = "./cache/dataset"
#tensor_directory_path = "./cache/tensor"
#download_directory_path = "/media/austin/extradrive1/cache/dataset"
#tensor_directory_path = "/media/austin/extradrive1/cache/tensor"
download_directory_path = "C:/cache/dataset"
tensor_directory_path = "C:/cache/tensor"


def get_default_datasets() -> List[Dataset]:
	return [
		Nabeel965HandwrittenWordsDatasetKaggleDataset(),
		IrisSklearnDataset()
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

			time.sleep(0.1)

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

		delete_directory_contents(
			directory_path=download_directory_path
		)
		delete_directory_contents(
			directory_path=tensor_directory_path
		)

		for dataset in get_default_datasets():

			dataset.download_to_directory(
				directory_path=download_directory_path,
				is_forced=False
			)

		datasets = get_default_datasets()

		tensor_cache = TensorCache(
			cache_directory_path=tensor_directory_path
		)

		for _ in range(len(datasets)**2):

			random.shuffle(datasets)
			tensor_cache.clear()

			for dataset in datasets:

				dataset.convert_to_tensor(
					download_directory_path=download_directory_path,
					tensor_cache_directory_path=tensor_directory_path
				)

				tensor_cache_category_sets = dataset.get_tensor_cache_category_sets(
					tensor_cache_directory_path=tensor_directory_path
				)

				self.assertNotEqual(0, len(tensor_cache_category_sets))

	def backup_test_download_birds(self):

		dataset = Gpiosenka100BirdSpeciesSharpenKaggleDataset()

		dataset.download_to_directory(
			directory_path=download_directory_path,
			is_forced=False
		)

	def backup_test_tensor_cache_birds(self):

		dataset = Gpiosenka100BirdSpeciesSharpenKaggleDataset()

		dataset.convert_to_tensor(
			download_directory_path=download_directory_path,
			tensor_cache_directory_path=tensor_directory_path
		)

# NOTE: consider simply adding any new dataset to the top of this file
