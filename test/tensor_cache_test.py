from __future__ import annotations
import unittest
from typing import List, Tuple, Dict
import tempfile
import torch
import numpy as np
import os
import kaggle
import PIL.Image
from src.austin_heller_repo.machine_learning.framework import TensorCache, TensorCacheCategorySet, TensorCacheCategorySubset, TensorCacheCategorySubsetCycle, TensorCacheCategorySubsetCycleRunner, TensorCacheCategorySubsetSequence, TensorCacheElement, TensorCacheElementLookup, get_float_tensor_from_image, ModuleTrainer, Conv2dToLinearToRecurrentToLinearModule, get_index_from_character, get_character_from_index, CharacterSetEnum, CharacterIndexOutOfRangeException, CharacterNotFoundException, english_character_set
from austin_heller_repo.common import StringEnum


class TensorCacheTest(unittest.TestCase):

	@classmethod
	def setUpClass(cls) -> None:

		if not os.path.exists("./cache"):
			raise Exception(f"Failed to find cache directory in test directory.")
		else:
			expected_dataset_names = [
				"nabeel965/handwritten-words-dataset"
			]
			missing_dataset_directory_path_per_dataset_name = {}  # type: Dict[str, str]
			for dataset_name in expected_dataset_names:
				dataset_directory_path = os.path.join("./cache/dataset", dataset_name)
				if not os.path.exists(dataset_directory_path):
					missing_dataset_directory_path_per_dataset_name[dataset_name] = dataset_directory_path
			if missing_dataset_directory_path_per_dataset_name:
				kaggle.api.authenticate()
				for dataset_name, dataset_directory_path in missing_dataset_directory_path_per_dataset_name.items():
					kaggle.api.dataset_download_files(
						dataset=dataset_name,
						path=dataset_directory_path,
						unzip=True
					)

	def test_character_to_index_and_back_utf8(self):

		main_string = "abcdefghijklmnopqrstuvwxyz0123456789-=+_.,<>`~'\"/?\\|[]{} \t\r\n\u7684"
		for character in main_string:
			index = get_index_from_character(
				character=character,
				character_set=CharacterSetEnum.Utf8
			)
			print(f"{character}: {index}")
			actual_character = get_character_from_index(
				index=index,
				character_set=CharacterSetEnum.Utf8
			)
			self.assertEqual(character, actual_character)

	def test_character_to_index_unexpected_character(self):

		with self.assertRaises(CharacterNotFoundException):
			get_index_from_character(
				character="\u7684",
				character_set=CharacterSetEnum.English
			)

	def test_index_to_character_unexpected_character_index(self):

		get_character_from_index(
			index=94,
			character_set=CharacterSetEnum.English
		)
		with self.assertRaises(CharacterIndexOutOfRangeException):
			get_character_from_index(
				index=95,
				character_set=CharacterSetEnum.English
			)

	def test_initialize(self):

		temp_directory = tempfile.TemporaryDirectory()

		try:
			tensor_cache = TensorCache(
				cache_directory_path=temp_directory.name
			)

			self.assertIsNotNone(tensor_cache)
		finally:
			temp_directory.cleanup()

	def test_create_tensor_cache_category_sets(self):

		temp_directory = tempfile.TemporaryDirectory()

		try:
			tensor_cache = TensorCache(
				cache_directory_path=temp_directory.name
			)

			tensor_cache_category_set = tensor_cache.create_tensor_cache_category_set(
				name="cats vs dogs",
				input_tensor_size=(20, 10),
				output_tensor_size=(1,)
			)

			cat_tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
				name="cat",
				output_tensor=torch.FloatTensor([0])
			)

			cat_tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=torch.FloatTensor(np.random.random((20, 10)))
			)
			cat_tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=torch.FloatTensor(np.random.random((20, 10)))
			)

			dog_tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
				name="dog",
				output_tensor=torch.FloatTensor([1])
			)
			dog_tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=torch.FloatTensor(np.random.random((20, 10)))
			)
			dog_tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=torch.FloatTensor(np.random.random((20, 10)))
			)

		finally:
			temp_directory.cleanup()

	def test_pulling_from_tensor_cache(self):

		temp_directory = tempfile.TemporaryDirectory()
		tensor_cache_category_subset_cycle_runner = None  # type: TensorCacheCategorySubsetCycleRunner

		try:
			tensor_cache = TensorCache(
				cache_directory_path=temp_directory.name
			)

			tensor_cache_category_set = tensor_cache.create_tensor_cache_category_set(
				name="cats vs dogs",
				input_tensor_size=(20, 10),
				output_tensor_size=(1,)
			)

			cat_tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
				name="cat",
				output_tensor=torch.FloatTensor([0])
			)

			cat_tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=torch.FloatTensor(np.random.random((20, 10)))
			)
			cat_tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=torch.FloatTensor(np.random.random((20, 10)))
			)

			dog_tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
				name="dog",
				output_tensor=torch.FloatTensor([1])
			)
			dog_tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=torch.FloatTensor(np.random.random((20, 10)))
			)
			dog_tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=torch.FloatTensor(np.random.random((20, 10)))
			)

			tensor_cache_category_subset_cycle = TensorCacheCategorySubsetCycle(
				tensor_cache_category_subset_sequences=[
					TensorCacheCategorySubsetSequence(
						tensor_cache_category_subsets=[
							tensor_cache_category_set.get_tensor_cache_category_subset_by_name(
								name="cat"
							)
						]
					),
					TensorCacheCategorySubsetSequence(
						tensor_cache_category_subsets=[
							tensor_cache_category_set.get_tensor_cache_category_subset_by_name(
								name="dog"
							)
						]
					)
				]
			)

			tensor_cache_category_subset_cycle_runner = TensorCacheCategorySubsetCycleRunner(
				name="test_runner",
				tensor_cache_category_subset_cycle=tensor_cache_category_subset_cycle,
				cache_length=100,
				is_cuda=False,
				is_decaching=False
			)

			index = 0
			is_successful = True
			while is_successful:
				is_successful, tensor_cache_element = tensor_cache_category_subset_cycle_runner.try_get_next_tensor_cache_element()
				if is_successful:
					print(f"{index}")
					#print(f"\t{_tensor_cache_element_lookup.get_tensor_cache_category_subset().get_name()}")
					#print(f"\t{_tensor_cache_element_lookup.get_index()}")
					print(f"\t{tensor_cache_element.get_input_tensor()[0]}")
					print(f"\t{tensor_cache_element.get_output_tensor()[0]}")
					index += 1

		finally:
			temp_directory.cleanup()
			if tensor_cache_category_subset_cycle_runner is not None:
				tensor_cache_category_subset_cycle_runner.dispose()

	def test_setup_tensor_cache_from_kaggle_dataset(self):

		dataset_directory_path = os.path.join("./cache/dataset/nabeel965/handwritten-words-dataset/data")
		tensor_cache_directory_path = os.path.join("./cache/tensor")

		tensor_cache = TensorCache(
			cache_directory_path=tensor_cache_directory_path
		)

		tensor_cache.clear()

		tensor_cache_category_set = tensor_cache.create_tensor_cache_category_set(
			name="nabeel965/handwritten-words-dataset",
			input_tensor_size=(146, 80),
			output_tensor_size=(None,)
		)

		capital_boxing_tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
			name="BOXING",
			output_tensor=torch.FloatTensor([get_index_from_character(
				character=x,
				character_set=CharacterSetEnum.English
			) for x in "BOXING"])
		)

		capital_boxing_directory_path = os.path.join(dataset_directory_path, "Capital", "BOXING", "BOXING")
		for file_name in os.listdir(capital_boxing_directory_path):
			file_path = os.path.join(capital_boxing_directory_path, file_name)
			image = PIL.Image.open(file_path)
			image_tensor = get_float_tensor_from_image(
				image=image
			)
			capital_boxing_tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=image_tensor
			)

		boxing_tensor_cache_category_subset = tensor_cache_category_set.get_tensor_cache_category_subset_by_name(
			name="BOXING"
		)

		self.assertIsNotNone(boxing_tensor_cache_category_subset)

		tensor_cache.clear()

	def test_module_trainer_training(self):

		dataset_directory_path = os.path.join("./cache/dataset/nabeel965/handwritten-words-dataset/data")
		tensor_cache_directory_path = os.path.join("./cache/tensor")

		tensor_cache = TensorCache(
			cache_directory_path=tensor_cache_directory_path
		)

		tensor_cache.clear()

		tensor_cache_category_set = tensor_cache.create_tensor_cache_category_set(
			name="nabeel965/handwritten-words-dataset",
			input_tensor_size=(146, 80),
			output_tensor_size=(None,)
		)

		word_directory_path_per_word = {}  # type: Dict[str, str]
		for subset_name in ["Capital", "small"]:
			for directory_name in os.listdir(os.path.join(dataset_directory_path, subset_name)):
				word_directory_path = os.path.join(dataset_directory_path, subset_name, directory_name, directory_name)
				word_directory_path_per_word[directory_name] = word_directory_path

		for word, word_directory_path in word_directory_path_per_word.items():
			capital_boxing_tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
				name=word,
				output_tensor=torch.LongTensor([
					get_index_from_character(
						character=x,
						character_set=CharacterSetEnum.English
					) for x in word
				] + [0])
			)

			for file_name in os.listdir(word_directory_path):
				file_path = os.path.join(word_directory_path, file_name)
				image = PIL.Image.open(file_path)
				image_tensor = get_float_tensor_from_image(
					image=image
				)
				capital_boxing_tensor_cache_category_subset.create_tensor_cache_element_input(
					input_tensor=image_tensor
				)

		module_trainer = ModuleTrainer(
			module=Conv2dToLinearToRecurrentToLinearModule(
				input_size=(146, 80),
				channels_total=1,
				input_to_flat_layers_total=10,
				flat_to_recurrent_lengths=[
					len(english_character_set)
				],
				recurrent_hidden_length=len(english_character_set),
				recurrent_to_output_lengths=[
					len(english_character_set)
				]
			),
			is_categorical=True,
			is_recurrent=True,
			is_cuda=False
		)

		module_input = TensorCacheCategorySubsetCycleRunner(
			name="test words",
			tensor_cache_category_subset_cycle=tensor_cache_category_set.get_tensor_cache_category_subset_cycle(),
			cache_length=1000,
			is_cuda=False,
			is_decaching=False,
			maximum_elements_total=10
		)

		try:

			module_trainer.train(
				module_input=module_input,
				learn_rate=0.1,
				maximum_batch_size=1
			)

		finally:
			module_input.dispose()
			tensor_cache.clear()

	def test_module_trainer_testing(self):

		dataset_directory_path = os.path.join("./cache/dataset/nabeel965/handwritten-words-dataset/data")
		tensor_cache_directory_path = os.path.join("./cache/tensor")

		tensor_cache = TensorCache(
			cache_directory_path=tensor_cache_directory_path
		)

		tensor_cache.clear()

		tensor_cache_category_set = tensor_cache.create_tensor_cache_category_set(
			name="nabeel965/handwritten-words-dataset",
			input_tensor_size=(146, 80),
			output_tensor_size=(None,)
		)

		word_directory_path_per_word = {}  # type: Dict[str, str]
		for subset_name in ["Capital", "small"]:
			for directory_name in os.listdir(os.path.join(dataset_directory_path, subset_name)):
				word_directory_path = os.path.join(dataset_directory_path, subset_name, directory_name, directory_name)
				word_directory_path_per_word[directory_name] = word_directory_path

		for word, word_directory_path in word_directory_path_per_word.items():
			capital_boxing_tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
				name=word,
				output_tensor=torch.LongTensor([
					get_index_from_character(
						character=x,
						character_set=CharacterSetEnum.English
					) for x in word
				] + [0])
			)

			for file_name in os.listdir(word_directory_path):
				file_path = os.path.join(word_directory_path, file_name)
				image = PIL.Image.open(file_path)
				image_tensor = get_float_tensor_from_image(
					image=image
				)
				capital_boxing_tensor_cache_category_subset.create_tensor_cache_element_input(
					input_tensor=image_tensor
				)

		module_trainer = ModuleTrainer(
			module=Conv2dToLinearToRecurrentToLinearModule(
				input_size=(146, 80),
				channels_total=1,
				input_to_flat_layers_total=10,
				flat_to_recurrent_lengths=[
					len(english_character_set),
					len(english_character_set)
				],
				recurrent_hidden_length=len(english_character_set) * 2,
				recurrent_to_output_lengths=[
					len(english_character_set),
					len(english_character_set)
				]
			),
			is_categorical=True,
			is_recurrent=True,
			is_cuda=False
		)

		module_input = TensorCacheCategorySubsetCycleRunner(
			name="test words",
			tensor_cache_category_subset_cycle=tensor_cache_category_set.get_tensor_cache_category_subset_cycle(),
			cache_length=1000,
			is_cuda=False,
			is_decaching=False,
			maximum_elements_total=10
		)

		try:

			module_trainer.test(
				module_input=module_input
			)

		finally:
			module_input.dispose()
			tensor_cache.clear()

	def test_module_trainer_training_and_testing(self):

		dataset_directory_path = os.path.join("./cache/dataset/nabeel965/handwritten-words-dataset/data")
		tensor_cache_directory_path = os.path.join("./cache/tensor")

		tensor_cache = TensorCache(
			cache_directory_path=tensor_cache_directory_path
		)

		tensor_cache.clear()

		tensor_cache_category_set = tensor_cache.create_tensor_cache_category_set(
			name="nabeel965/handwritten-words-dataset",
			input_tensor_size=(146, 80),
			output_tensor_size=(None,)
		)

		word_directory_path_per_word = {}  # type: Dict[str, str]
		for subset_name in ["Capital", "small"]:
			for directory_name in os.listdir(os.path.join(dataset_directory_path, subset_name)):
				word_directory_path = os.path.join(dataset_directory_path, subset_name, directory_name, directory_name)
				word_directory_path_per_word[directory_name] = word_directory_path

		for word, word_directory_path in word_directory_path_per_word.items():
			capital_boxing_tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
				name=word,
				output_tensor=torch.LongTensor([
					get_index_from_character(
						character=x,
						character_set=CharacterSetEnum.English
					) for x in word
				] + [0])
			)

			for file_name in os.listdir(word_directory_path):
				file_path = os.path.join(word_directory_path, file_name)
				image = PIL.Image.open(file_path)
				image_tensor = get_float_tensor_from_image(
					image=image
				)
				capital_boxing_tensor_cache_category_subset.create_tensor_cache_element_input(
					input_tensor=image_tensor
				)

		module_trainer = ModuleTrainer(
			module=Conv2dToLinearToRecurrentToLinearModule(
				input_size=(146, 80),
				channels_total=1,
				input_to_flat_layers_total=10,
				flat_to_recurrent_lengths=[
					len(english_character_set) * 1,
					len(english_character_set) * 1
				],
				recurrent_hidden_length=len(english_character_set) * 2,
				recurrent_to_output_lengths=[
					len(english_character_set) * 2,
					len(english_character_set)
				]
			),
			is_categorical=True,
			is_recurrent=True,
			is_cuda=False
		)

		module_input = TensorCacheCategorySubsetCycleRunner(
			name="test words",
			tensor_cache_category_subset_cycle=tensor_cache_category_set.get_tensor_cache_category_subset_cycle(),
			cache_length=1000,
			is_cuda=False,
			is_decaching=False,
			maximum_elements_total=1
		)

		try:

			module_trainer.train(
				module_input=module_input,
				learn_rate=0.001,
				maximum_batch_size=1,
				epochs=100
			)

			module_trainer.test(
				module_input=module_input
			)

		finally:
			module_input.dispose()
			tensor_cache.clear()
