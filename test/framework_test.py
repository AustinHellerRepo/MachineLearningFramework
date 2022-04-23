from __future__ import annotations
import unittest
import os
import pathlib
import torch
from src.austin_heller_repo.machine_learning.framework import ModuleTrainer, TensorCache, TensorCacheCategorySubsetCycleRunner, SpecificLinearToLinearModule, LinearToConv2dModule, Conv2dToLinearToConv2dModule
from src.austin_heller_repo.machine_learning.dataset.sklearn.iris_dataset import IrisSklearnDataset


download_directory_path = "./cache/framework/dataset"
tensor_cache_directory_path = "./cache/framework/tensor"


def get_iris_tensor_cache_category_subset_cycle_runner() -> TensorCacheCategorySubsetCycleRunner:

	tensor_cache = TensorCache(
		cache_directory_path=tensor_cache_directory_path
	)

	tensor_cache.clear()

	dataset = IrisSklearnDataset()
	dataset.download_to_directory(
		directory_path=download_directory_path,
		is_forced=False
	)
	dataset.convert_to_tensor(
		download_directory_path=download_directory_path,
		tensor_cache_directory_path=tensor_cache_directory_path
	)

	tensor_cache_category_sets = tensor_cache.get_tensor_cache_category_sets(
		name="iris",
		input_tensor_size=None,
		output_tensor_size=None
	)
	return TensorCacheCategorySubsetCycleRunner(
		name="iris",
		tensor_cache_category_subset_cycle=tensor_cache_category_sets[0].get_tensor_cache_category_subset_cycle(),
		cache_length=150,
		is_cuda=False,
		is_decaching=False
	)


class IrisModule(torch.nn.Module):

	def __init__(self):
		super().__init__()

		self.__linear_layer = SpecificLinearToLinearModule(
			layer_sizes=(
				4,
				5,
				3
			)
		)
		self.__softmax = torch.nn.Softmax(
			dim=1
		)

	def forward(self, module_input: torch.FloatTensor):
		#return self.__softmax(self.__linear_layer(module_input))
		return self.__linear_layer(module_input)


class FrameworkTest(unittest.TestCase):

	@classmethod
	def setUpClass(cls) -> None:
		pathlib.Path(tensor_cache_directory_path).mkdir(parents=True, exist_ok=True)

	def test_train_specific_linear_to_linear(self):

		module_trainer = ModuleTrainer(
			module=IrisModule(),
			is_categorical=True,
			is_recurrent=False,
			is_cuda=False
		)

		cycle_runner = get_iris_tensor_cache_category_subset_cycle_runner()

		try:
			module_trainer.train(
				module_input=cycle_runner,
				learn_rate=0.01,
				maximum_batch_size=10,
				epochs=5
			)

			cycle_runner.reset()

			module_trainer.test(
				module_input=cycle_runner
			)

		finally:
			cycle_runner.dispose()

	def test_linear_to_conv2d_module(self):

		module = LinearToConv2dModule(
			input_size=4,
			layer_total=4,
			module_output_size=(100, 100),
			output_channel_total=3
		)

		for _ in range(100):
			module_input = torch.rand((4,))
			module_output = module(module_input)

			self.assertEqual((3, 100, 100), module_output.shape)

	def test_conv2d_to_linear_to_conv2d_module(self):

		module = Conv2dToLinearToConv2dModule(
			input_size=(200, 100),
			input_channels=1,
			input_to_linear_layer_total=4,
			linear_layer_lengths=(100, 200, 300),
			linear_to_output_layer_total=4,
			output_size=(200, 100),
			output_channels=3
		)

		for _ in range(10):
			module_input = torch.rand((1, 100, 200))
			module_output = module(module_input)
			self.assertEqual((3, 100, 200), module_output.shape)
