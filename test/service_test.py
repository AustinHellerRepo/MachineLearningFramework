from __future__ import annotations
import unittest
from typing import List, Tuple, Dict, Type
import uuid
import time
import torch
from src.austin_heller_repo.machine_learning.common import FloatTensorModuleInput, LongTensorModuleOutput
from src.austin_heller_repo.machine_learning.service import ServiceStructure, ServiceClientServerMessage, ServiceClientServerMessageTypeEnum, ServiceSourceTypeEnum, TrainingDataPurposeTypeEnum, ClientStructure
from src.austin_heller_repo.machine_learning.framework import TensorCache, ModuleTrainer, TensorCacheCategorySubsetCycleRunner, Conv2dToLinearToRecurrentToLinearModule, english_character_set, TensorCacheElement
from src.austin_heller_repo.machine_learning.dataset.dataset import Dataset
from src.austin_heller_repo.machine_learning.dataset.kaggle.nabeel965_handwritten_words_dataset import Nabeel965HandwrittenWordsDatasetKaggleDataset
from src.austin_heller_repo.machine_learning.implementation.tensor_cache_module_service import TensorCacheModuleServiceStructureFactory
from austin_heller_repo.socket_queued_message_framework import ServerMessenger, ClientMessenger, HostPointer, ServerSocketFactory, StructureFactory, ClientSocketFactory, Structure, ClientMessengerFactory
from austin_heller_repo.common import delete_directory_contents


def get_default_datasets() -> List[Dataset]:
	return [
		Nabeel965HandwrittenWordsDatasetKaggleDataset()
	]


dataset_directory_path = "./cache/dataset"
tensor_directory_path = "./cache/tensor"


client_port = 39132


def get_default_server_messenger() -> ServerMessenger:
	return ServerMessenger(
		server_socket_factory_and_local_host_pointer_per_source_type={
			ServiceSourceTypeEnum.Client: (
				ServerSocketFactory(),
				HostPointer(
					host_address="localhost",
					host_port=client_port
				)
			)
		},
		client_server_message_class=ServiceClientServerMessage,
		source_type_enum_class=ServiceSourceTypeEnum,
		server_messenger_source_type=ServiceSourceTypeEnum.Service,
		structure_factory=TensorCacheModuleServiceStructureFactory(
			tensor_cache=TensorCache(
				cache_directory_path="./cache/tensor"
			),
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
			is_cuda=False,
			cache_length=1000,
			is_decaching=False,
			learning_rate=0.001,
			maximum_batch_size=1,
			delay_between_training_seconds_total=1.0,
			is_debug=False
		)
	)


def get_default_client_structure() -> ClientStructure:
	return ClientStructure(
		service_client_messenger_factory=ClientMessengerFactory(
			client_socket_factory=ClientSocketFactory(),
			server_host_pointer=HostPointer(
				host_address="localhost",
				host_port=client_port
			),
			client_server_message_class=ServiceClientServerMessage,
			is_debug=False
		)
	)


class ServiceTest(unittest.TestCase):

	def test_initialize(self):

		server_messenger = get_default_server_messenger()

		try:
			self.assertIsNotNone(server_messenger)

			server_messenger.start_receiving_from_clients()

			time.sleep(1.0)

			server_messenger.stop_receiving_from_clients()

		finally:
			server_messenger.dispose()

	def test_send_dataset(self):

		server_messenger = get_default_server_messenger()

		category_set_name = f"test category set name: {uuid.uuid4()}"
		category_subset_name = f"test category subset name: {uuid.uuid4()}"

		try:
			self.assertIsNotNone(server_messenger)

			server_messenger.start_receiving_from_clients()

			client_structure = get_default_client_structure()

			for dataset in get_default_datasets():

				delete_directory_contents(
					directory_path=dataset_directory_path
				)
				delete_directory_contents(
					directory_path=tensor_directory_path
				)

				dataset.download_to_directory(
					directory_path=dataset_directory_path,
					is_forced=False
				)

				dataset.convert_to_tensor(
					download_directory_path=dataset_directory_path,
					tensor_cache_directory_path=tensor_directory_path
				)

				tensor_cache_category_set = dataset.get_tensor_cache_category_set(
					tensor_cache_directory_path=tensor_directory_path
				)

				tensor_cache_category_subset_cycle = tensor_cache_category_set.get_tensor_cache_category_subset_cycle()

				tensor_cache_category_subset_cycle.reset()

				is_successful, tensor_cache_element_lookup = tensor_cache_category_subset_cycle.try_get_next_tensor_cache_element_lookup()
				index = 0

				while is_successful:
					tensor_cache_element = TensorCacheElement.get_tensor_cache_element_from_tensor_cache_element_lookup(
						tensor_cache_element_lookup=tensor_cache_element_lookup,
						is_cuda=False
					)

					input_tensor = tensor_cache_element.get_input_tensor()
					if isinstance(input_tensor, torch.FloatTensor):
						module_input = FloatTensorModuleInput.get_from_float_tensor(
							float_tensor=input_tensor
						)
					else:
						raise Exception(f"Unexpected input tensor type: {type(input_tensor)}.")

					output_tensor = tensor_cache_element.get_output_tensor()
					if isinstance(output_tensor, torch.FloatTensor):
						module_output = FloatTensorModuleInput.get_from_float_tensor(
							float_tensor=output_tensor
						)
					elif isinstance(output_tensor, torch.LongTensor):
						module_output = LongTensorModuleOutput.get_from_long_tensor(
							long_tensor=output_tensor
						)
					else:
						raise Exception(f"Unexpected output tensor type: {type(output_tensor)}.")

					training_data_purpose_type = TrainingDataPurposeTypeEnum.Validation if (index + 1) % 10 == 0 else TrainingDataPurposeTypeEnum.Training
					client_structure.add_training_data_announcement(
						category_set_name=category_set_name,
						category_subset_name=category_subset_name,
						module_input=module_input,
						module_output=module_output,
						training_data_purpose_type=training_data_purpose_type
					)

					is_successful, tensor_cache_element_lookup = tensor_cache_category_subset_cycle.try_get_next_tensor_cache_element_lookup()
					index += 1

			server_messenger.stop_receiving_from_clients()

		finally:
			server_messenger.dispose()