from __future__ import annotations
from typing import List, Tuple, Dict, TextIO, Type
import torch
from decimal import Decimal, getcontext
import numpy as np
import math
import random
import time
import os
from ast import literal_eval
import PIL.Image
from austin_heller_repo.threading import start_thread, Semaphore
from austin_heller_repo.common import get_unique_directory_path, get_unique_file_path, ElapsedTimerMessageManager, delete_directory_contents, StringEnum


class CharacterSetEnum(StringEnum):
	English = "english",
	Utf8 = "utf8"


english_character_set = ["END_OF_WORD", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", ".", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "/", "\\", "<", ">", ",", "?", ":", ";", "'", '"', "[", "]", "{", "}", "_", "+", "=", "|", "~", " "]


class CharacterNotFoundException(Exception):

	def __init__(self, *, character: str, character_set: CharacterSetEnum):
		super().__init__(f"Character \"{character}\" not found in character set {character_set}.")

		self.__character = character
		self.__character_set = character_set

	def get_character(self) -> str:
		return self.__character

	def get_character_set(self) -> CharacterSetEnum:
		return self.__character_set


class CharacterIndexOutOfRangeException(Exception):

	def __init__(self, *, index: int, character_set: CharacterSetEnum):
		super().__init__(f"Character index \"{index}\" out of range for character set {character_set}.")

		self.__index = index
		self.__character_set = character_set

	def get_index(self) -> int:
		return self.__index

	def get_character_set(self) -> CharacterSetEnum:
		return self.__character_set


def get_index_from_character(*, character: str, character_set: CharacterSetEnum) -> int:
	if len(character) != 1:
		raise Exception(f"Unexpected number of characters: {len(character)}")
	else:
		if character_set == CharacterSetEnum.Utf8:
			index = int.from_bytes(character[0].encode(encoding="UTF-8"), "big")
		elif character_set == CharacterSetEnum.English:
			if character not in english_character_set:
				raise CharacterNotFoundException(
					character=character,
					character_set=character_set
				)
			index = english_character_set.index(character)
		else:
			raise NotImplementedError(f"CharacterSet not implemented: {character_set}")
		return index


def get_character_from_index(*, index: int, character_set: CharacterSetEnum) -> str:
	if character_set == CharacterSetEnum.Utf8:
		characters = index.to_bytes(4, "big").decode(encoding="UTF-8")
		while len(characters) > 1:
			if characters[0] == "\x00":
				characters = characters[1:]
			else:
				break
		character = characters
	elif character_set == CharacterSetEnum.English:
		if index < len(english_character_set):
			character = english_character_set[index]
		else:
			raise CharacterIndexOutOfRangeException(
				index=index,
				character_set=character_set
			)
	else:
		raise NotImplementedError(f"CharacterSet not implemented: {character_set}")
	return character


class SpecificLinearToLinearModule(torch.nn.Module):

	def __init__(self, layer_sizes: List[int]):
		torch.nn.Module.__init__(self)

		self.__layer_sizes = layer_sizes

		self.__batchnorm_module_list = None  # type: torch.nn.ModuleList
		self.__layer_module_list = None  # type: torch.nn.ModuleList

		self.__initialize()

	def __initialize(self):
		batchnorm_modules = []  # type: List[torch.nn.BatchNorm1d]
		layer_modules = []  # type: List[torch.nn.Linear]
		for layer_index in range(len(self.__layer_sizes) - 1):
			batchnorm = torch.nn.BatchNorm1d(self.__layer_sizes[layer_index])
			batchnorm_modules.append(batchnorm)
			linear = torch.nn.Linear(self.__layer_sizes[layer_index], self.__layer_sizes[layer_index + 1])
			layer_modules.append(linear)
		self.__batchnorm_module_list = torch.nn.ModuleList(batchnorm_modules)
		self.__layer_module_list = torch.nn.ModuleList(layer_modules)

	def forward(self, module_input: torch.FloatTensor):

		module_output = module_input
		for module_index, (batchnorm_module, layer_module) in enumerate(zip(self.__batchnorm_module_list, self.__layer_module_list)):
			if module_index != 0:
				module_output = torch.relu(module_output)
			#if len(module_output.shape) > 1 and module_output.shape[0] != 1:
			if not self.training or len(module_output.shape) > 1 and module_output.shape[0] != 1:
				module_output = batchnorm_module(module_output)

			if False:
				linear_module_outputs = []  # type: List[torch.FloatTensor]
				for linear_module_output_index in range(module_output.shape[0]):
					linear_module_output = layer_module(module_output[linear_module_output_index])
					linear_module_outputs.append(linear_module_output)
				module_output = torch.stack(_linear_module_outputs)
			else:
				module_output = layer_module(module_output)

		return module_output


class AligningConv2dModule(torch.nn.Module):

	def __init__(self, module_input_size: Tuple[int, int], layer_total: int, input_channel_total: int, output_channel_total: int):
		torch.nn.Module.__init__(self)

		self.__module_input_size = module_input_size
		self.__layer_total = layer_total
		self.__input_channel_total = input_channel_total
		self.__output_channel_total = output_channel_total

		self.__conv2d_layers = None  # type: torch.nn.ModuleList
		self.__relu_layers = None  # type: torch.nn.ModuleList
		self.__batchnorm_layers = None  # type: torch.nn.ModuleList
		self.__linear_output_length = None  # type: int

		self.__initialize()

	def __initialize(self):

		expected_output_sizes = self._get_expected_output_sizes()
		expected_output_sizes.insert(0, self.__module_input_size)
		previous_output_size = expected_output_sizes.pop()

		stride_and_kernel_width_and_height_pairs = []  # type: List[Tuple[Tuple[int, int], Tuple[int, int]]]
		possible_stride_and_kernel_width_pairs = []  # type: List[Tuple[int, int]]
		possible_stride_and_kernel_height_pairs = []  # type: List[Tuple[int, int]]
		for expected_output_size in reversed(expected_output_sizes):

			possible_stride_and_kernel_width_pairs.clear()
			current_stride_width = 1
			while True:
				current_kernel_width = expected_output_size[0] - current_stride_width * (previous_output_size[0] - 1)
				if current_kernel_width >= current_stride_width * 2 or (current_kernel_width == 1 and current_stride_width == 1):
					possible_stride_and_kernel_width_pairs.append((current_stride_width, current_kernel_width))
					if previous_output_size[0] == 1:
						break
				else:
					break
				current_stride_width += 1

			possible_stride_and_kernel_height_pairs.clear()
			current_stride_height = 1
			while True:
				current_kernel_height = expected_output_size[1] - current_stride_height * (previous_output_size[1] - 1)
				if current_kernel_height >= current_stride_height * 2 or (current_kernel_height == 1 and current_stride_height == 1):
					possible_stride_and_kernel_height_pairs.append((current_stride_height, current_kernel_height))
					if previous_output_size[1] == 1:
						break
				else:
					break
				current_stride_height += 1

			stride_and_kernel_width_and_height_pairs.append((possible_stride_and_kernel_width_pairs[-1], possible_stride_and_kernel_height_pairs[-1]))

			previous_output_size = expected_output_size

		stride_and_kernel_width_and_height_pairs.reverse()

		batchnorm_layers = []  # type: List[torch.nn.BatchNorm2d]
		conv2d_layers = []  # type: List[torch.nn.Conv2d]
		relu_layers = []  # type: List[torch.nn.ReLU]
		channels = list(np.linspace(self.__input_channel_total, self.__output_channel_total, num=self.__layer_total + 1))
		for layer_index in range(self.__layer_total):
			previous_channel = int(channels[layer_index])
			next_channel = int(channels[layer_index + 1])

			batchnorm = torch.nn.BatchNorm2d(previous_channel)

			kernel = (stride_and_kernel_width_and_height_pairs[layer_index][0][1], stride_and_kernel_width_and_height_pairs[layer_index][1][1])  # type: Tuple[int, int]
			stride = (stride_and_kernel_width_and_height_pairs[layer_index][0][0], stride_and_kernel_width_and_height_pairs[layer_index][1][0])  # type: Tuple[int, int]

			conv2d = torch.nn.Conv2d(in_channels=previous_channel, out_channels=next_channel, kernel_size=kernel, stride=stride)
			relu = torch.nn.ReLU()
			batchnorm_layers.append(batchnorm)
			conv2d_layers.append(conv2d)
			relu_layers.append(relu)

		self.__batchnorm_layers = torch.nn.ModuleList(batchnorm_layers)
		self.__conv2d_layers = torch.nn.ModuleList(conv2d_layers)
		self.__relu_layers = torch.nn.ModuleList(relu_layers)

	def _get_expected_output_sizes(self) -> List[Tuple[int, int]]:
		getcontext().prec = 100
		min_exponential_constant_width = None
		max_exponential_constant_width = Decimal(1.0)
		min_exponential_constant_height = None
		max_exponential_constant_height = Decimal(1.0)
		two = Decimal(2.0)

		found_width = False
		current_estimated_expected_output_widths = []  # type: List[int]
		while not found_width:
			current_estimated_expected_output_widths.clear()
			if self.__layer_total >= self.__module_input_size[0]:
				for output_width in np.linspace(self.__module_input_size[0], 1, self.__layer_total):
					current_estimated_expected_output_widths.append(int(output_width))
				found_width = True
			else:
				previous_output_width = self.__module_input_size[0]
				current_exponential_constant_width = (max_exponential_constant_width if min_exponential_constant_width is None else (max_exponential_constant_width + min_exponential_constant_width) / two)
				for layer_index in range(self.__layer_total):
					next_output_width = math.floor(previous_output_width / pow(current_exponential_constant_width, Decimal(layer_index))) - 1
					current_estimated_expected_output_widths.append(next_output_width)
					previous_output_width = next_output_width
				if current_estimated_expected_output_widths[-1] > 1:
					if min_exponential_constant_width is None:
						max_exponential_constant_width *= two
					else:
						min_exponential_constant_width = current_exponential_constant_width
				elif current_estimated_expected_output_widths[-1] < 1:
					if min_exponential_constant_width is None:
						min_exponential_constant_width = current_exponential_constant_width / two
					else:
						max_exponential_constant_width = current_exponential_constant_width
				else:
					found_width = True

		found_height = False
		current_estimated_expected_output_heights = []  # type: List[int]
		while not found_height:
			current_estimated_expected_output_heights.clear()
			if self.__layer_total >= self.__module_input_size[1]:
				for output_height in np.linspace(self.__module_input_size[1], 1, self.__layer_total):
					current_estimated_expected_output_heights.append(int(output_height))
				found_height = True
			else:
				previous_output_height = self.__module_input_size[1]
				current_exponential_constant_height = (max_exponential_constant_height if min_exponential_constant_height is None else (max_exponential_constant_height + min_exponential_constant_height) / two)
				for layer_index in range(self.__layer_total):
					next_output_height = math.floor(previous_output_height / pow(current_exponential_constant_height, Decimal(layer_index))) - 1
					current_estimated_expected_output_heights.append(next_output_height)
					previous_output_height = next_output_height
				if current_estimated_expected_output_heights[-1] > 1:
					if min_exponential_constant_height is None:
						max_exponential_constant_height *= two
					else:
						min_exponential_constant_height = current_exponential_constant_height
				elif current_estimated_expected_output_heights[-1] < 1:
					if min_exponential_constant_height is None:
						min_exponential_constant_height = current_exponential_constant_height / two
					else:
						max_exponential_constant_height = current_exponential_constant_height
				else:
					found_height = True

		current_estimated_expected_output_sizes = []  # type: List[Tuple[int, int]]
		for width, height in zip(current_estimated_expected_output_widths, current_estimated_expected_output_heights):
			current_estimated_expected_output_sizes.append((width, height))

		return current_estimated_expected_output_sizes

	def forward(self, module_input: torch.FloatTensor):

		#module_output = module_input.view(1, 1, module_input.shape[0], module_input.shape[1])
		module_output = module_input.view(module_input.shape[0], self.__input_channel_total, module_input.shape[2], module_input.shape[1])
		for conv2d_module, relu_module, batchnorm_module in zip(self.__conv2d_layers, self.__relu_layers, self.__batchnorm_layers):
			module_output = batchnorm_module(module_output)
			module_output = conv2d_module(module_output)
			module_output = relu_module(module_output)
		module_output = module_output.squeeze()
		if len(module_output.shape) == 1:
			module_output = torch.unsqueeze(module_output, dim=0)
		return module_output


class Conv2dToLinearToRecurrentToLinearModule(torch.nn.Module):

	def __init__(self, *, input_size: Tuple[int, int], channels_total: int, input_to_flat_layers_total: int, flat_to_recurrent_lengths: List[int], recurrent_hidden_length: int, recurrent_to_output_lengths: List[int]):
		super().__init__()

		self.__input_size = input_size
		self.__channels_total = channels_total
		self.__input_to_flat_layers_total = input_to_flat_layers_total
		self.__flat_to_recurrent_lengths = flat_to_recurrent_lengths
		self.__recurrent_hidden_length = recurrent_hidden_length
		self.__recurrent_to_output_lengths = recurrent_to_output_lengths

		self.__conv2d_module = None  # type: AligningConv2dModule
		self.__conv2d_to_recurrent_linear_module = None  # type: SpecificLinearToLinearModule
		self.__recurrent_module = None  # type: torch.nn.GRUCell
		self.__recurrent_to_output_linear_module = None  # type: SpecificLinearToLinearModule

		self.__initialize()

	def __initialize(self):

		if self.__recurrent_hidden_length != self.__recurrent_to_output_lengths[0]:
			raise Exception(f"The recurrent_hidden_length needs to equal the first layer length of the recurrent_to_output_lengths.")

		self.__conv2d_module = AligningConv2dModule(
			module_input_size=self.__input_size,
			layer_total=self.__input_to_flat_layers_total,
			input_channel_total=self.__channels_total,
			output_channel_total=self.__flat_to_recurrent_lengths[0]
		)

		self.__conv2d_to_recurrent_linear_module = SpecificLinearToLinearModule(
			layer_sizes=self.__flat_to_recurrent_lengths
		)

		if False:
			self.__recurrent_module = torch.nn.GRUCell(
				input_size=self.__flat_to_recurrent_lengths[-1],
				hidden_size=self.__recurrent_hidden_length
			)
		elif False:
			self.__recurrent_module = torch.nn.GRU(
				input_size=self.__flat_to_recurrent_lengths[-1],
				hidden_size=self.__recurrent_hidden_length
			)
		elif True:
			self.__recurrent_module = torch.nn.RNN(
				input_size=self.__flat_to_recurrent_lengths[-1],
				hidden_size=self.__recurrent_hidden_length
			)
		elif True:
			self.__recurrent_module = torch.nn.RNNCell(
				input_size=self.__flat_to_recurrent_lengths[-1],
				hidden_size=self.__recurrent_hidden_length
			)

		self.__recurrent_to_output_linear_module = SpecificLinearToLinearModule(
			layer_sizes=self.__recurrent_to_output_lengths
		)

	def forward(self, module_input: torch.FloatTensor, hidden_input: torch.FloatTensor = None):

		batch_size = module_input.shape[0]
		sequence_length = module_input.shape[1]
		image_width = module_input.shape[2]
		image_height = module_input.shape[3]

		module_outputs = []  # type: List[torch.FloatTensor]
		hidden_outputs = []  # type: List[torch.FloatTensor]
		for batch_index in range(batch_size):

			if hidden_input is None:
				hidden_output = torch.FloatTensor(np.zeros((batch_size, self.__recurrent_hidden_length)))
			else:
				hidden_output = hidden_input

			if isinstance(self.__recurrent_module, torch.nn.GRU):
				module_output = module_input[batch_index]
				module_output = self.__conv2d_module(module_output)
				module_output = self.__linear_module(module_output)
				for sequence_index in range(sequence_length):
					hidden_output = self.__recurrent_module(module_output[sequence_index].view(1, -1), hidden_output)
				module_outputs.append(hidden_output)
			elif isinstance(self.__recurrent_module, torch.nn.RNN) or isinstance(self.__recurrent_module, torch.nn.RNNCell):
				module_output = module_input[batch_index]
				module_output = self.__conv2d_module(module_output)
				module_output = self.__conv2d_to_recurrent_linear_module(module_output)
				for sequence_index in range(sequence_length):
					transformed_module_input = module_output[sequence_index].view(1, -1)
					final_output, hidden_output = self.__recurrent_module(transformed_module_input, hidden_output)
				final_output = self.__recurrent_to_output_linear_module(final_output)
				module_outputs.append(final_output)
				hidden_outputs.append(hidden_output)
		return torch.stack(module_outputs).squeeze(dim=0), torch.stack(hidden_outputs).squeeze(dim=0)


def get_float_tensor_from_image(image: PIL.Image.Image):
	image_array = np.array(image)
	image_array_32 = image_array.astype(np.float32)
	image_tensor = torch.tensor(image_array_32) / 255.0
	image_tensor_shaped = image_tensor.unsqueeze(0)
	return image_tensor_shaped


class TensorCacheCategorySubsetCycleRunner():

	def __init__(self, *, name: str, tensor_cache_category_subset_cycle: TensorCacheCategorySubsetCycle, cache_length: int, is_cuda: bool, is_decaching: bool, maximum_elements_total: int = None):

		self.__name = name
		self.__tensor_cache_category_subset_cycle = tensor_cache_category_subset_cycle
		self.__cache_length = cache_length
		self.__is_cuda = is_cuda
		self.__is_decaching = is_decaching
		self.__maximum_elements_total = maximum_elements_total

		self.__tensor_cache_element_lookups = []  # type: List[TensorCacheElementLookup]
		self.__tensor_cache_element_lookups_total = None  # type: int
		self.__next_cache_index = 0
		self.__last_cache_index = 0

		self.__tensor_cache_element_per_index = {}  # type: Dict[int, TensorCacheElement]
		self.__polling_thread = None
		self.__polling_thread_semaphore = None  # type: Semaphore
		self.__polling_thread_is_running = False
		self.__precache_time_sleep_seconds = 0.44
		self.__precache_thread = None
		self.__precache_thread_semaphore = None  # type: Semaphore
		self.__precache_thread_is_running = False
		self.__decache_thread = None
		self.__decache_thread_semaphore = None  # type: Semaphore
		self.__decache_thread_is_running = False
		self.__last_decached_index = None  # type: int

		self.__initialize()

	def __initialize(self):

		self.__decache_thread_semaphore = Semaphore()

		self.__tensor_cache_category_subset_cycle.reset()

		is_successful, tensor_cache_element_lookup = self.__tensor_cache_category_subset_cycle.try_get_next_tensor_cache_element_lookup()

		if not is_successful:

			self.__tensor_cache_element_lookups_total = 0

		else:

			while is_successful:
				self.__tensor_cache_element_lookups.append(tensor_cache_element_lookup)
				is_successful, tensor_cache_element_lookup = self.__tensor_cache_category_subset_cycle.try_get_next_tensor_cache_element_lookup()

			self.__tensor_cache_element_lookups_total = len(self.__tensor_cache_element_lookups)
			random.shuffle(self.__tensor_cache_element_lookups)

			self.__polling_thread_is_running = True

			def polling_thread_method():
				start_seconds = time.perf_counter()
				while self.__polling_thread_is_running:
					tick_seconds = time.perf_counter()
					print(f"{self.__name}: {int(tick_seconds - start_seconds)}: {self.__last_cache_index / self.__tensor_cache_element_lookups_total}: self.__last_decached_index: {self.__last_decached_index}, self.__next_cache_index: {self.__next_cache_index}, self.__last_cache_index: {self.__last_cache_index}, difference: {self.__last_cache_index - self.__next_cache_index}, sleep: {self.__precache_time_sleep_seconds}")
					time.sleep(1)

			self.__polling_thread = start_thread(target=polling_thread_method)

			self.__precache_thread_is_running = True

			def precache_thread_method():
				while self.__precache_thread_is_running:
					time.sleep(self.__precache_time_sleep_seconds)
					if self.__last_cache_index - self.__next_cache_index > self.__cache_length / 2.0:
						self.__precache_time_sleep_seconds += 0.01
					else:
						self.__precache_time_sleep_seconds = max(0.01, self.__precache_time_sleep_seconds - 0.01)
					#time.sleep(0.44)
					self.__decache_thread_semaphore.acquire()
					while self.__last_cache_index < self.__tensor_cache_element_lookups_total and self.__last_cache_index - self.__next_cache_index < self.__cache_length:
						tensor_cache_element_lookup = self.__tensor_cache_element_lookups[self.__last_cache_index]
						tensor_cache_element = TensorCacheElement.get_tensor_cache_element_from_tensor_cache_element_lookup(
							tensor_cache_element_lookup=tensor_cache_element_lookup,
							is_cuda=self.__is_cuda
						)
						self.__tensor_cache_element_per_index[self.__last_cache_index] = tensor_cache_element
						self.__last_cache_index += 1
					self.__decache_thread_semaphore.release()

			self.__precache_thread = start_thread(target=precache_thread_method)

			self.__decache_thread_is_running = True

			def decache_thread_method():
				while self.__decache_thread_is_running:
					time.sleep(1)
					self.__decache_thread_semaphore.acquire()
					found_cached_tensor_cache_element = True
					current_index = self.__next_cache_index - 1
					while found_cached_tensor_cache_element:
						if current_index not in self.__tensor_cache_element_per_index.keys():
							found_cached_tensor_cache_element = False
							self.__last_decached_index = current_index
						else:
							del self.__tensor_cache_element_per_index[current_index]
							current_index -= 1
					self.__decache_thread_semaphore.release()

			if self.__is_decaching:
				self.__decache_thread = start_thread(target=decache_thread_method)

	def reset(self):
		# decache existing elements
		self.__decache_thread_semaphore.acquire()
		while len(self.__tensor_cache_element_per_index.keys()) != 0:
			index = next(iter(self.__tensor_cache_element_per_index.keys()))
			del self.__tensor_cache_element_per_index[index]
		self.__decache_thread_semaphore.release()

		# reset index pointers
		self.__next_cache_index = 0
		self.__last_cache_index = 0

	def try_get_next_tensor_cache_element(self) -> Tuple[bool, TensorCacheElement]:
		if self.__next_cache_index == self.__tensor_cache_element_lookups_total or \
				(self.__maximum_elements_total is not None and self.__next_cache_index == self.__maximum_elements_total):
			return False, None
		else:
			while self.__next_cache_index == self.__last_cache_index:
				time.sleep(0.001)
			tensor_cache_element = self.__tensor_cache_element_per_index[self.__next_cache_index]
			#del self.__tensor_cache_element_per_index[self.__next_cache_index]
			self.__next_cache_index += 1
			return True, tensor_cache_element

	def try_get_next_tensor_cache_element_batch(self, maximum_batch_size: int) -> Tuple[bool, torch.Tensor, torch.Tensor]:
		tensor_cache_elements = []  # type: List[TensorCacheElement]
		tensor_cache_elements_total = 0
		is_successful = True
		while is_successful and tensor_cache_elements_total < maximum_batch_size:
			is_successful, tensor_cache_element = self.try_get_next_tensor_cache_element()
			if is_successful:
				tensor_cache_elements.append(tensor_cache_element)
				tensor_cache_elements_total += 1

		input_tensors = []  # type: List[torch.Tensor]
		output_tensors = []  # type: List[torch.Tensor]
		for tensor_cache_element in tensor_cache_elements:
			input_tensor = tensor_cache_element.get_input_tensor()
			output_tensor = tensor_cache_element.get_output_tensor()
			input_tensors.append(input_tensor)
			output_tensors.append(output_tensor)

		if len(input_tensors) != 0:
			batch_input_tensor = torch.stack(input_tensors)
			batch_output_tensor = torch.stack(output_tensors)
		else:
			batch_input_tensor = None
			batch_output_tensor = None

		return is_successful, batch_input_tensor, batch_output_tensor

	def dispose(self):
		self.__polling_thread_is_running = False
		self.__precache_thread_is_running = False
		self.__decache_thread_is_running = False


class TensorCache():

	def __init__(self, *, cache_directory_path: str):

		self.__cache_directory_path = cache_directory_path

	def clear(self):

		delete_directory_contents(
			directory_path=self.__cache_directory_path
		)

	def create_tensor_cache_category_set(self, *, name: str, input_tensor_size: Tuple[int, ...], output_tensor_size: Tuple[int, ...]) -> TensorCacheCategorySet:

		if self.get_tensor_cache_category_set(
			name=name,
			input_tensor_size=input_tensor_size,
			output_tensor_size=output_tensor_size
		) is not None:

			raise Exception(f"TensorCache with same name and sizes already exists.")

		tensor_cache_category_set_directory_path = get_unique_directory_path(
			parent_directory_path=self.__cache_directory_path
		)
		os.makedirs(tensor_cache_category_set_directory_path)
		tensor_cache_category_set_index_file_path = os.path.join(tensor_cache_category_set_directory_path, "index.idx")
		with open(tensor_cache_category_set_index_file_path, "w") as file_handle:
			pass

		tensor_cache_category_set = TensorCacheCategorySet(
			name=name,
			cache_directory_path=tensor_cache_category_set_directory_path,
			input_tensor_size=input_tensor_size,
			output_tensor_size=output_tensor_size
		)
		index_file_path = os.path.join(self.__cache_directory_path, "index.idx")
		with open(index_file_path, "a") as file_handle:
			tensor_cache_category_set.append_to_index_file_handle(file_handle)
		return tensor_cache_category_set

	def get_tensor_cache_category_set(self, *, name: str, input_tensor_size: Tuple[int, ...], output_tensor_size: Tuple[int, ...]) -> TensorCacheCategorySet:

		tensor_cache_category_set = None  # type: TensorCacheCategorySet
		index_file_path = os.path.join(self.__cache_directory_path, "index.idx")

		if os.path.exists(index_file_path):
			with open(index_file_path, "r") as file_handle:
				while tensor_cache_category_set is None:
					is_successful, tensor_cache_category_set = TensorCacheCategorySet.try_read_from_index_file_handle(file_handle)
					if is_successful:
						if not (tensor_cache_category_set.get_name() == name and
								tensor_cache_category_set.get_input_tensor_size() == input_tensor_size and
								tensor_cache_category_set.get_output_tensor_size() == output_tensor_size):
							tensor_cache_category_set = None
					else:
						break

		return tensor_cache_category_set

	def get_tensor_cache_category_subset_cycle(self) -> TensorCacheCategorySubsetCycle:

		tensor_cache_category_subset_cycle = None  # type: TensorCacheCategorySubsetCycle
		index_file_path = os.path.join(self.__cache_directory_path, "index.idx")

		tensor_cache_category_subsets = []  # type: List[TensorCacheCategorySubset]

		if os.path.exists(index_file_path):
			with open(index_file_path, "r") as file_handle:
				is_successful, tensor_cache_category_set = TensorCacheCategorySet.try_read_from_index_file_handle(file_handle)
				if is_successful:
					raise Exception(f"Failed to load tensor cache category set.")

				current_tensor_cache_category_subsets = tensor_cache_category_set.get_tensor_cache_category_subsets()
				tensor_cache_category_subsets.extend(current_tensor_cache_category_subsets)

		tensor_cache_category_subset_cycle = TensorCacheCategorySubsetCycle(
			tensor_cache_category_subset_sequences=[
				TensorCacheCategorySubsetSequence(
					tensor_cache_category_subsets=tensor_cache_category_subsets
				)
			]
		)

		return tensor_cache_category_subset_cycle


class TensorCacheCategorySet():

	"""
		Contains a reference to a directory that contains sub-directories for each possible output type
	"""

	def __init__(self, *, name: str, cache_directory_path: str, input_tensor_size: Tuple[int, ...], output_tensor_size: Tuple[int, ...]):

		self.__name = name
		self.__cache_directory_path = cache_directory_path
		self.__input_tensor_size = input_tensor_size
		self.__output_tensor_size = output_tensor_size

		self.__tensor_cache_category_subsets = []  # type: List[TensorCacheCategorySubset]

		self.__initialize()

	def __initialize(self):

		index_file_path = os.path.join(self.__cache_directory_path, "index.idx")
		with open(index_file_path, "r") as file_handle:
			is_successful = True
			while is_successful:
				is_successful, tensor_cache_category_subset = TensorCacheCategorySubset.try_read_from_index_file_handle(file_handle)
				if is_successful:
					self.__tensor_cache_category_subsets.append(tensor_cache_category_subset)

	def get_name(self) -> str:
		return self.__name

	def get_cache_directory_path(self) -> str:
		return self.__cache_directory_path

	def get_input_tensor_size(self) -> Tuple[int, ...]:
		return self.__input_tensor_size

	def get_output_tensor_size(self) -> Tuple[int, ...]:
		return self.__output_tensor_size

	def get_tensor_cache_category_subsets(self) -> List[TensorCacheCategorySubset]:
		return self.__tensor_cache_category_subsets

	def create_tensor_cache_category_subset(self, *, name: str, output_tensor: torch.Tensor) -> TensorCacheCategorySubset:

		tensor_cache_category_subset_directory_path = get_unique_directory_path(
			parent_directory_path=self.__cache_directory_path
		)
		os.makedirs(tensor_cache_category_subset_directory_path)

		output_tensor_file_path = get_unique_file_path(
			parent_directory_path=self.__cache_directory_path,
			extension="tensor"
		)
		torch.save(output_tensor, output_tensor_file_path)

		tensor_cache_category_subset = TensorCacheCategorySubset(
			name=name,
			cache_directory_path=tensor_cache_category_subset_directory_path,
			output_tensor_file_path=output_tensor_file_path
		)

		index_file_path = os.path.join(self.__cache_directory_path, "index.idx")
		with open(index_file_path, "a") as file_handle:
			tensor_cache_category_subset.append_to_index_file_handle(file_handle)

		self.__tensor_cache_category_subsets.append(tensor_cache_category_subset)
		return tensor_cache_category_subset

	def append_to_index_file_handle(self, file_handle: TextIO):
		file_handle.writelines([
			self.__name,
			"\n",
			self.__cache_directory_path,
			"\n",
			str(self.__input_tensor_size),
			"\n",
			str(self.__output_tensor_size),
			"\n"
		])

	@staticmethod
	def try_read_from_index_file_handle(file_handle: TextIO) -> Tuple[bool, TensorCacheCategorySet]:
		name = file_handle.readline().strip()
		if name == "":
			return False, None
		cache_directory_path = file_handle.readline().strip()
		input_tensor_size = literal_eval(file_handle.readline().strip())
		output_tensor_size = literal_eval(file_handle.readline().strip())
		tensor_cache_category_set = TensorCacheCategorySet(
			name=name,
			cache_directory_path=cache_directory_path,
			input_tensor_size=input_tensor_size,
			output_tensor_size=output_tensor_size
		)
		return True, tensor_cache_category_set

	def get_tensor_cache_category_subset_by_name(self, name: str) -> TensorCacheCategorySubset:
		specific_tensor_cache_category_subsets = []
		for tensor_cache_category_subset in self.__tensor_cache_category_subsets:
			if tensor_cache_category_subset.get_name() == name:
				specific_tensor_cache_category_subset = tensor_cache_category_subset
				specific_tensor_cache_category_subsets.append(specific_tensor_cache_category_subset)
		specific_tensor_cache_category_subsets_total = len(specific_tensor_cache_category_subsets)
		if specific_tensor_cache_category_subsets_total > 1:
			raise Exception(f"Found more than one tensor cache category subset with name \"{name}\".")
		elif specific_tensor_cache_category_subsets_total == 0:
			return None
		else:
			return specific_tensor_cache_category_subsets[0]

	def get_tensor_cache_category_subset_cycle(self) -> TensorCacheCategorySubsetCycle:

		tensor_cache_category_subset_sequences = []  # type: List[TensorCacheCategorySubsetSequence]

		for tensor_cache_category_subset in self.__tensor_cache_category_subsets:
			tensor_cache_category_subset_sequence = TensorCacheCategorySubsetSequence(
				tensor_cache_category_subsets=[
					tensor_cache_category_subset
				]
			)
			tensor_cache_category_subset_sequences.append(tensor_cache_category_subset_sequence)

		tensor_cache_category_subset_cycle = TensorCacheCategorySubsetCycle(
			tensor_cache_category_subset_sequences=tensor_cache_category_subset_sequences
		)

		return tensor_cache_category_subset_cycle

	def precache(self, tensor_cache: TensorCache):
		for tensor_cache_category_subset in self.__tensor_cache_category_subsets:
			is_successful = True
			while is_successful:
				is_successful, tensor_cache_element_lookup = tensor_cache_category_subset.try_get_next_tensor_cache_element_lookup()
				if is_successful:
					tensor_cache.get_tensor_cache_element(
						tensor_cache_element_lookup=tensor_cache_element_lookup
					)


class TensorCacheCategorySubset():

	"""
		Contains a reference to all of the possible inputs for the specific output
	"""

	def __init__(self, *, name: str, cache_directory_path: str, output_tensor_file_path: str):

		self.__name = name
		self.__cache_directory_path = cache_directory_path
		self.__output_tensor_file_path = output_tensor_file_path

		self.__cache_element_index = None  # type: int
		self.__cache_element_cycle_total = 0  # type: int
		self.__cache_element_file_paths = []  # type: List[str]

		self.__initialize()

	def __initialize(self):

		for file_name in os.listdir(self.__cache_directory_path):
			file_path = os.path.join(self.__cache_directory_path, file_name)
			self.__cache_element_file_paths.append(file_path)

	def get_name(self) -> str:
		return self.__name

	def get_cache_directory_path(self) -> str:
		return self.__cache_directory_path

	def get_output_tensor_file_path(self) -> str:
		return self.__output_tensor_file_path

	def create_tensor_cache_element_input(self, input_tensor: torch.Tensor):

		input_tensor_file_path = get_unique_file_path(
			parent_directory_path=self.__cache_directory_path,
			extension="tensor"
		)
		torch.save(input_tensor, input_tensor_file_path)
		self.__cache_element_file_paths.append(input_tensor_file_path)

	def append_to_index_file_handle(self, file_handle: TextIO):
		file_handle.writelines([
			self.__name,
			"\n",
			self.__cache_directory_path,
			"\n",
			self.__output_tensor_file_path,
			"\n",
		])

	@staticmethod
	def try_read_from_index_file_handle(file_handle: TextIO) -> Tuple[bool, TensorCacheCategorySubset]:
		name = file_handle.readline().strip()
		if name == "":
			return False, None
		cache_directory_path = file_handle.readline().strip()
		output_tensor_file_path = file_handle.readline().strip()
		tensor_cache_category_subset = TensorCacheCategorySubset(
			name=name,
			cache_directory_path=cache_directory_path,
			output_tensor_file_path=output_tensor_file_path
		)
		return True, tensor_cache_category_subset

	def reset(self):
		self.__cache_element_index = None

	def try_get_next_tensor_cache_element_lookup(self) -> Tuple[bool, TensorCacheElementLookup or None]:

		if self.__cache_element_index is None:
			self.__cache_element_index = 0
		else:
			if self.__cache_element_index < len(self.__cache_element_file_paths):
				self.__cache_element_index += 1
				if self.__cache_element_index == len(self.__cache_element_file_paths):
					return False, None
			else:
				return False, None
		tensor_cache_element_lookup = TensorCacheElementLookup(
			input_file_path=self.__cache_element_file_paths[self.__cache_element_index],
			output_file_path=self.__output_tensor_file_path,
			tensor_cache_category_subset=self,
			index=self.__cache_element_index
		)
		return True, tensor_cache_element_lookup

	def is_completed(self) -> bool:
		return self.__cache_element_index is not None and self.__cache_element_index + 1 == len(self.__cache_element_file_paths)


class TensorCacheCategorySubsetSequence():

	def __init__(self, tensor_cache_category_subsets: List[TensorCacheCategorySubset]):

		self.__tensor_cache_category_subsets = tensor_cache_category_subsets

		self.__tensor_cache_category_subsets_index = None

	def reset(self):
		for tensor_cache_category_subset in self.__tensor_cache_category_subsets:
			tensor_cache_category_subset.reset()
		self.__tensor_cache_category_subsets_index = None

	def try_get_next_tensor_cache_element_lookup(self) -> Tuple[bool, TensorCacheElementLookup or None]:

		if self.__tensor_cache_category_subsets_index is None:
			self.__tensor_cache_category_subsets_index = 0
		if self.__tensor_cache_category_subsets_index < len(self.__tensor_cache_category_subsets):
			tensor_cache_element_lookup = None
			while tensor_cache_element_lookup is None:
				is_successful, tensor_cache_element_lookup = self.__tensor_cache_category_subsets[self.__tensor_cache_category_subsets_index].try_get_next_tensor_cache_element_lookup()
				if not is_successful:
					self.__tensor_cache_category_subsets_index += 1
					if self.__tensor_cache_category_subsets_index == len(self.__tensor_cache_category_subsets):
						return False, None
		else:
			return False, None  # there are no subsets in this sequence

		return True, tensor_cache_element_lookup

	def is_empty(self) -> bool:
		return len(self.__tensor_cache_category_subsets) == 0

	def is_completed(self) -> bool:
		return self.__tensor_cache_category_subsets_index is not None and \
			   self.__tensor_cache_category_subsets_index + 1 == len(self.__tensor_cache_category_subsets) and \
				self.__tensor_cache_category_subsets[self.__tensor_cache_category_subsets_index].is_completed()


class TensorCacheCategorySubsetCycle():

	def __init__(self, tensor_cache_category_subset_sequences: List[TensorCacheCategorySubsetSequence]):

		self.__tensor_cache_category_subset_sequences = tensor_cache_category_subset_sequences

		self.__tensor_cache_category_subset_sequence_index = None
		self.__tensor_cache_category_subset_sequence_completed = []  # type: List[bool]

		self.__initialize()

	def __initialize(self):

		for index in range(len(self.__tensor_cache_category_subset_sequences)):
			self.__tensor_cache_category_subset_sequence_completed.append(False)

	def reset(self):
		for index in range(len(self.__tensor_cache_category_subset_sequence_completed)):
			self.__tensor_cache_category_subset_sequence_completed[index] = False
			self.__tensor_cache_category_subset_sequences[index].reset()
		self.__tensor_cache_category_subset_sequence_index = None

	def try_get_next_tensor_cache_element_lookup(self) -> Tuple[bool, TensorCacheElementLookup]:

		if all([x.is_empty() for x in self.__tensor_cache_category_subset_sequences]):
			return False, None
		if all(self.__tensor_cache_category_subset_sequence_completed) and self.__tensor_cache_category_subset_sequence_index == 0:
			return False, None
		if self.__tensor_cache_category_subset_sequence_index is None:
			self.__tensor_cache_category_subset_sequence_index = 0
		tensor_cache_element_lookup = None

		while tensor_cache_element_lookup is None:
			is_successful, tensor_cache_element_lookup = self.__tensor_cache_category_subset_sequences[self.__tensor_cache_category_subset_sequence_index].try_get_next_tensor_cache_element_lookup()
			if not is_successful:
				raise Exception("Unexpected pull after ending sequence")
			if self.__tensor_cache_category_subset_sequences[self.__tensor_cache_category_subset_sequence_index].is_completed():
				self.__tensor_cache_category_subset_sequence_completed[self.__tensor_cache_category_subset_sequence_index] = True
				self.__tensor_cache_category_subset_sequences[self.__tensor_cache_category_subset_sequence_index].reset()
			self.__tensor_cache_category_subset_sequence_index += 1
			if self.__tensor_cache_category_subset_sequence_index == len(self.__tensor_cache_category_subset_sequences):
				self.__tensor_cache_category_subset_sequence_index = 0
		return True, tensor_cache_element_lookup


class TensorCacheElementLookup():

	def __init__(self, input_file_path: str, output_file_path: str, tensor_cache_category_subset: TensorCacheCategorySubset, index: int):

		self.__input_file_path = input_file_path
		self.__output_file_path = output_file_path
		self.__tensor_cache_category_subset = tensor_cache_category_subset
		self.__index = index

	def __eq__(self, other):
		if not isinstance(other, TensorCacheElementLookup):
			return False
		else:
			if self.__input_file_path == other.__input_file_path and \
				self.__output_file_path == other.__output_file_path:
				return True
			else:
				return False

	def __hash__(self):
		return hash(self.__input_file_path + self.__output_file_path)

	def get_input_file_path(self) -> str:
		return self.__input_file_path

	def get_output_file_path(self) -> str:
		return self.__output_file_path

	def get_tensor_cache_category_subset(self) -> TensorCacheCategorySubset:
		return self.__tensor_cache_category_subset

	def get_index(self) -> int:
		return self.__index


class TensorCacheElement():

	def __init__(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, tensor_cache_element_lookup: TensorCacheElementLookup):

		self.__input_tensor = input_tensor
		self.__output_tensor = output_tensor
		self.__tensor_cache_element_lookup = tensor_cache_element_lookup

	def get_input_tensor(self) -> torch.Tensor:
		return self.__input_tensor

	def get_output_tensor(self) -> torch.Tensor:
		return self.__output_tensor

	def get_tensor_cache_element_lookup(self) -> TensorCacheElementLookup:
		return self.__tensor_cache_element_lookup

	@staticmethod
	def get_tensor_cache_element_from_tensor_cache_element_lookup(*, tensor_cache_element_lookup: TensorCacheElementLookup, is_cuda: bool) -> TensorCacheElement:

		input_tensor = torch.load(tensor_cache_element_lookup.get_input_file_path())
		if is_cuda:
			input_tensor = input_tensor.cuda()
		output_tensor = torch.load(tensor_cache_element_lookup.get_output_file_path())
		if is_cuda:
			output_tensor = output_tensor.cuda()

		tensor_cache_element = TensorCacheElement(
			input_tensor=input_tensor,
			output_tensor=output_tensor,
			tensor_cache_element_lookup=tensor_cache_element_lookup
		)
		return tensor_cache_element


class ModuleTrainer():

	def __init__(self, *, module: torch.nn.Module, is_categorical: bool, is_recurrent: bool, is_cuda: bool, is_debug: bool = False):

		self.__module = module
		self.__is_categorical = is_categorical
		self.__is_recurrent = is_recurrent
		self.__is_cuda = is_cuda
		self.__is_debug = is_debug

		self.__elapsed_timer_message_manager = None  # type: ElapsedTimerMessageManager
		self.__loss = None

		self.__initialize()

	def __initialize(self):

		self.__elapsed_timer_message_manager = ElapsedTimerMessageManager(
			include_datetime_prefix=True,
			include_stack=True
		)

	def __log(self, message: str):

		if self.__is_debug:
			self.__elapsed_timer_message_manager.print(
				message=message,
				override_stack_offset=1
			)

	def train(self, *, module_input: TensorCacheCategorySubsetCycleRunner, learn_rate: float, maximum_batch_size: int, epochs: int):

		self.__log("start")

		if self.__is_categorical:
			criterion = torch.nn.CrossEntropyLoss()
		else:
			criterion = torch.nn.MSELoss()

		optimizer = torch.optim.SGD(self.__module.parameters(), lr=learn_rate, momentum=0.9)

		self.__log("Setup criterion and optimizers")

		total_loss = 0

		for epoch_index in range(epochs):

			self.__log("Epoch started")

			module_input.reset()

			self.__log("Reset cycle runner")

			module_input_index = 0
			is_successful = True
			while is_successful:

				self.__log("Starting loop")

				is_successful, batch_input_tensor, batch_output_tensor = module_input.try_get_next_tensor_cache_element_batch(
					maximum_batch_size=maximum_batch_size if not self.__is_recurrent else 1
				)

				self.__log("Pulled tensor cache element")

				#while is_successful and module_input_index < 100:
				if is_successful:

					optimizer.zero_grad()

					self.__log("Set zero grad")

					if self.__is_recurrent:

						hidden_output = None
						loss = None
						expected_output = batch_output_tensor.squeeze(dim=0)
						for expected_output_element in expected_output:

							module_output, hidden_output = self.__module(batch_input_tensor, hidden_output)

							if self.__is_categorical:
								module_output = module_output.view(1, -1)

							self.__log("Collected encoder output")

							if loss is None:
								loss = criterion(module_output, expected_output_element.view(-1))
							else:
								loss += criterion(module_output, expected_output_element.view(-1))
						loss_scalar = 1.0 / len(expected_output)
					else:
						raise NotImplementedError()

					loss.backward()

					optimizer.step()

					self.__log("Performed loss backward and optimizer steps")

					module_input_index += 1

					if not self.__is_cuda:
						current_loss = loss.item() * loss_scalar
						total_loss += current_loss

						self.__log("Appended total loss")

						if module_input_index % 1 == 0:
							print(f"{module_input_index}: {current_loss}")

					if module_input_index % 100 == 0:
						elapsed_seconds_total_per_message = self.__elapsed_timer_message_manager.get_elapsed_seconds_total_per_message()
						for message_to_message in elapsed_seconds_total_per_message.keys():
							print(f"{message_to_message}: {elapsed_seconds_total_per_message[message_to_message]}")

			self.__log("Ended loop")

		if not self.__is_cuda:
			if self.__loss is None:
				self.__loss = total_loss
			else:
				self.__loss += total_loss

			self.__log("Appended total loss")

	def test(self, *, module_input: TensorCacheCategorySubsetCycleRunner):

		self.__module.eval()

		module_input.reset()

		categorical_errors_per_output_category_per_expected_category = {}  # type: Dict[int, Dict[int, int]]
		categorical_successes_total = 0

		is_successful = True
		while is_successful:
			is_successful, batch_input_tensor, batch_output_tensor = module_input.try_get_next_tensor_cache_element_batch(
				maximum_batch_size=1
			)
			if is_successful:

				# show output error
				if self.__is_categorical:

					if self.__is_recurrent:
						hidden_output = None

						for batch_output_tensor_element in batch_output_tensor.squeeze():

							module_output, hidden_output = self.__module(batch_input_tensor, hidden_output)

							expected_value = batch_output_tensor_element.item()

							softmax = torch.exp(module_output)
							index = softmax.argmax()
							output_value = index.item()
							if output_value != expected_value:
								if expected_value not in categorical_errors_per_output_category_per_expected_category.keys():
									categorical_errors_per_output_category_per_expected_category[expected_value] = {}
								if output_value not in categorical_errors_per_output_category_per_expected_category[expected_value].keys():
									categorical_errors_per_output_category_per_expected_category[expected_value][output_value] = 1
								else:
									categorical_errors_per_output_category_per_expected_category[expected_value][output_value] += 1
							else:
								categorical_successes_total += 1
					else:
						raise NotImplementedError()
				else:
					raise NotImplementedError()

		# print errors
		if self.__is_categorical:
			print(f"Success: {categorical_successes_total}")
			for expected_value in categorical_errors_per_output_category_per_expected_category.keys():
				for output_value in categorical_errors_per_output_category_per_expected_category[expected_value].keys():
					print(f"Expected {expected_value}, found {output_value}: {categorical_errors_per_output_category_per_expected_category[expected_value][output_value]}")
		else:
			raise NotImplementedError()


