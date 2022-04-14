from __future__ import annotations
import torch
import copy
import uuid
from datetime import datetime
from austin_heller_repo.threading import Semaphore
from austin_heller_repo.socket_queued_message_framework import StructureFactory
from src.austin_heller_repo.machine_learning.common import ModuleInput, ModuleOutput, FloatTensorModuleInput, LongTensorModuleOutput, FloatTensorModuleOutput
from src.austin_heller_repo.machine_learning.service import ServiceStructure, TrainingDataPurposeTypeEnum
from src.austin_heller_repo.machine_learning.framework import TensorCache, TensorCacheCategorySubsetCycleRunner, ModuleTrainer


class TensorCacheModuleServiceStructure(ServiceStructure):

	def __init__(self, *, tensor_cache: TensorCache, module: torch.nn.Module, is_categorical: bool, is_recurrent: bool, is_cuda: bool, cache_length: int, is_decaching: bool, learning_rate: float, maximum_batch_size: int, delay_between_training_seconds_total: float, is_debug: bool = False):

		self.__tensor_cache = tensor_cache
		self.__module = module
		self.__is_categorical = is_categorical
		self.__is_recurrent = is_recurrent
		self.__is_cuda = is_cuda
		self.__cache_length = cache_length
		self.__is_decaching = is_decaching
		self.__learning_rate = learning_rate
		self.__maximum_batch_size = maximum_batch_size
		self.__is_debug = is_debug

		self.__detection_module = None  # type: torch.nn.Module
		self.__detection_module_semaphore = Semaphore()
		self.__tensor_cache_semaphore = Semaphore()

		self.__initialize()

		super().__init__(
			delay_between_training_seconds_total=delay_between_training_seconds_total,
			is_debug=is_debug
		)

	def __initialize(self):

		self.__copy_module_into_detection_module()

	def __copy_module_into_detection_module(self):
		self.__detection_module_semaphore.acquire()
		try:
			self.__detection_module = copy.deepcopy(self.__module)
		finally:
			self.__detection_module_semaphore.release()

	def get_module_output(self, *, module_input: ModuleInput) -> ModuleOutput:

		if isinstance(module_input, FloatTensorModuleInput):
			input_tensor = module_input.get_float_tensor()
		else:
			raise Exception(f"Unexpected module input type: {type(module_input)}.")

		self.__detection_module_semaphore.acquire()
		try:
			detection_module_output = self.__detection_module(input_tensor)
		finally:
			self.__detection_module_semaphore.release()

		if isinstance(detection_module_output, torch.FloatTensor):
			module_output = FloatTensorModuleOutput.get_from_float_tensor(
				float_tensor=detection_module_output
			)
		else:
			raise Exception(f"Unexpected module output type: {type(detection_module_output)}.")

		return module_output

	def add_training_data(self, *, category_set_name: str, category_subset_name: str, module_input: ModuleInput, module_output: ModuleOutput, purpose: TrainingDataPurposeTypeEnum):

		if isinstance(module_input, FloatTensorModuleInput):
			input_tensor = module_input.get_float_tensor()
		else:
			raise Exception(f"Unexpected module input type: {type(module_input)}.")

		if isinstance(module_output, FloatTensorModuleOutput):
			output_tensor = module_output.get_float_tensor()
		elif isinstance(module_output, LongTensorModuleOutput):
			output_tensor = module_output.get_long_tensor()
		else:
			raise Exception(f"Unexpected module output type: {type(module_input)}.")

		input_tensor_size = tuple(input_tensor.shape)
		output_tensor_size = tuple(output_tensor.shape)

		self.__tensor_cache_semaphore.acquire()
		try:
			tensor_cache_category_sets = self.__tensor_cache.get_tensor_cache_category_sets(
				name=category_set_name,
				input_tensor_size=input_tensor_size,
				output_tensor_size=output_tensor_size
			)

			if len(tensor_cache_category_sets) == 0:
				tensor_cache_category_set = self.__tensor_cache.create_tensor_cache_category_set(
					name=category_set_name,
					input_tensor_size=tuple(input_tensor.shape),
					output_tensor_size=tuple(output_tensor.shape)
				)
			else:
				tensor_cache_category_set = tensor_cache_category_sets[0]

			tensor_cache_category_subset = tensor_cache_category_set.get_tensor_cache_category_subset_by_name(
				name=category_subset_name
			)

			if tensor_cache_category_subset is None:
				tensor_cache_category_subset = tensor_cache_category_set.create_tensor_cache_category_subset(
					name=category_subset_name,
					output_tensor=output_tensor
				)

			tensor_cache_category_subset.create_tensor_cache_element_input(
				input_tensor=input_tensor
			)

		finally:
			self.__tensor_cache_semaphore.release()

	def train_module(self):

		module_trainer = ModuleTrainer(
			module=self.__module,
			is_categorical=self.__is_categorical,
			is_recurrent=self.__is_recurrent,
			is_cuda=self.__is_cuda,
			is_debug=self.__is_debug
		)

		self.__tensor_cache_semaphore.acquire()
		try:
			module_input = TensorCacheCategorySubsetCycleRunner(
				name=f"{TensorCacheModuleServiceStructure.__name__}: {str(uuid.uuid4())}",
				tensor_cache_category_subset_cycle=self.__tensor_cache.get_tensor_cache_category_subset_cycle(),
				cache_length=self.__cache_length,
				is_cuda=self.__is_cuda,
				is_decaching=self.__is_decaching
			)
		finally:
			self.__tensor_cache_semaphore.release()

		try:
			module_trainer.train(
				module_input=module_input,
				learn_rate=self.__learning_rate,
				maximum_batch_size=self.__maximum_batch_size,
				epochs=1
			)

			self.__copy_module_into_detection_module()

		finally:
			module_input.dispose()


class TensorCacheModuleServiceStructureFactory(StructureFactory):

	def __init__(self, *, tensor_cache: TensorCache, module: torch.nn.Module, is_categorical: bool, is_recurrent: bool, is_cuda: bool, cache_length: int, is_decaching: bool, learning_rate: float, maximum_batch_size: int, delay_between_training_seconds_total: float, is_debug: bool = False):

		self.__tensor_cache = tensor_cache
		self.__module = module
		self.__is_categorical = is_categorical
		self.__is_recurrent = is_recurrent
		self.__is_cuda = is_cuda
		self.__cache_length = cache_length
		self.__is_decaching = is_decaching
		self.__learning_rate = learning_rate
		self.__maximum_batch_size = maximum_batch_size
		self.__delay_between_training_seconds_total = delay_between_training_seconds_total
		self.__is_debug = is_debug

	def get_structure(self) -> TensorCacheModuleServiceStructure:
		return TensorCacheModuleServiceStructure(
			tensor_cache=self.__tensor_cache,
			module=self.__module,
			is_categorical=self.__is_categorical,
			is_recurrent=self.__is_recurrent,
			is_cuda=self.__is_cuda,
			cache_length=self.__cache_length,
			is_decaching=self.__is_decaching,
			learning_rate=self.__learning_rate,
			maximum_batch_size=self.__maximum_batch_size,
			delay_between_training_seconds_total=self.__delay_between_training_seconds_total,
			is_debug=self.__is_debug
		)
