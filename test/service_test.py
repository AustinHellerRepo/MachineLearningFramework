from __future__ import annotations
import unittest
from typing import List, Tuple, Dict, Type
from io import BytesIO
import uuid
import time
import torch
import os
from datetime import datetime
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import urllib.request
import random
import base64
from src.austin_heller_repo.machine_learning.common import FloatTensorModuleInput, LongTensorModuleOutput, ImageModuleInput, LocalizationListModuleOutput, LocalizationModuleOutput
from src.austin_heller_repo.machine_learning.service import ServiceStructure, ServiceClientServerMessage, ServiceClientServerMessageTypeEnum, ServiceSourceTypeEnum, TrainingDataPurposeTypeEnum, ClientStructure
from src.austin_heller_repo.machine_learning.framework import TensorCache, ModuleTrainer, TensorCacheCategorySubsetCycleRunner, Conv2dToLinearToRecurrentToLinearModule, english_character_set, TensorCacheElement
from src.austin_heller_repo.machine_learning.dataset.dataset import Dataset
from src.austin_heller_repo.machine_learning.dataset.kaggle.nabeel965_handwritten_words_dataset import Nabeel965HandwrittenWordsDatasetKaggleDataset
from src.austin_heller_repo.machine_learning.implementation.tensor_cache_module_service_structure import TensorCacheModuleServiceStructureFactory
from src.austin_heller_repo.machine_learning.implementation.yolov5_service_structure import YoloV5ServiceStructureFactory, YoloV5ModelTypeEnum
from austin_heller_repo.socket_queued_message_framework import ServerMessenger, ClientMessenger, HostPointer, ServerSocketFactory, StructureFactory, ClientSocketFactory, Structure, ClientMessengerFactory
from austin_heller_repo.common import delete_directory_contents, StringEnum, get_random_rainbow_color


is_delete_cache = False


def get_default_datasets() -> List[Dataset]:
	return [
		Nabeel965HandwrittenWordsDatasetKaggleDataset()
	]


def get_random_words() -> List[str]:
	url = "https://www.mit.edu/~ecprice/wordlist.10000"
	response = urllib.request.urlopen(url)
	txt = response.read()
	words_bytes = txt.splitlines()  # type: List[bytes]
	words = [x.decode() for x in words_bytes]
	return words


dataset_directory_path = "./cache/dataset"
tensor_directory_path = "./cache/tensor"
service_tensor_directory_path = "./cache/tensor_service"
git_clone_directory_path = "./cache/git"


client_port = 39132


class ServiceStructureTypeEnum(StringEnum):
	TensorCacheModuleServiceStructure = "tensor_cache_module_service_structure"
	YoloV5ServiceStructure = "yolov5_service_structure"


def get_default_server_messenger(*, service_structure_type: ServiceStructureTypeEnum) -> ServerMessenger:

	if service_structure_type == ServiceStructureTypeEnum.TensorCacheModuleServiceStructure:
		service_structure_factory = TensorCacheModuleServiceStructureFactory(
			tensor_cache=TensorCache(
				cache_directory_path=service_tensor_directory_path
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
	elif service_structure_type == ServiceStructureTypeEnum.YoloV5ServiceStructure:
		service_structure_factory = YoloV5ServiceStructureFactory(
			git_clone_directory_path=git_clone_directory_path,
			label_classes_total=2,
			is_cuda=False,
			delay_between_training_seconds_total=1.0,
			is_git_pull_forced=False,
			image_size=1024,
			training_batch_size=1,
			model_type=YoloV5ModelTypeEnum.YoloV5N,
			is_debug=False
		)
	else:
		raise NotImplementedError(f"{ServiceStructureTypeEnum.__name__} not implemented: {service_structure_type.value}.")

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
		structure_factory=service_structure_factory
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

	@classmethod
	def setUpClass(cls) -> None:
		global is_delete_cache

		if is_delete_cache:
			delete_directory_contents(
				directory_path="./cache"
			)

	def test_initialize(self):

		server_messenger = get_default_server_messenger(
			service_structure_type=ServiceStructureTypeEnum.TensorCacheModuleServiceStructure
		)

		try:
			self.assertIsNotNone(server_messenger)

			server_messenger.start_receiving_from_clients()

			time.sleep(1.0)

			server_messenger.stop_receiving_from_clients()

		finally:
			server_messenger.dispose()

	def test_connect_and_disconnect_client_to_service(self):

		server_messenger = get_default_server_messenger(
			service_structure_type=ServiceStructureTypeEnum.TensorCacheModuleServiceStructure
		)

		category_set_name = f"test category set name: {uuid.uuid4()}"
		category_subset_name = f"test category subset name: {uuid.uuid4()}"

		client_structure = None  # type: ClientStructure

		try:
			self.assertIsNotNone(server_messenger)

			server_messenger.start_receiving_from_clients()

			client_structure = get_default_client_structure()

			time.sleep(1.0)

			server_messenger.stop_receiving_from_clients()

		finally:
			server_messenger.dispose()
			if client_structure is not None:
				client_structure.dispose()

	def test_send_dataset(self):

		server_messenger = get_default_server_messenger(
			service_structure_type=ServiceStructureTypeEnum.TensorCacheModuleServiceStructure
		)

		client_structure = None  # type: ClientStructure

		try:
			self.assertIsNotNone(server_messenger)

			server_messenger.start_receiving_from_clients()

			client_structure = get_default_client_structure()

			for dataset in get_default_datasets():

				if os.path.exists(dataset_directory_path):
					delete_directory_contents(
						directory_path=dataset_directory_path
					)
				if os.path.exists(tensor_directory_path):
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

				tensor_cache_category_sets = dataset.get_tensor_cache_category_sets(
					tensor_cache_directory_path=tensor_directory_path
				)

				index = 0

				for tensor_cache_category_set in tensor_cache_category_sets:

					tensor_cache_category_subset_cycle = tensor_cache_category_set.get_tensor_cache_category_subset_cycle()

					tensor_cache_category_subset_cycle.reset()

					is_successful, tensor_cache_element_lookup = tensor_cache_category_subset_cycle.try_get_next_tensor_cache_element_lookup()

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
							category_set_name=tensor_cache_category_set.get_name(),
							category_subset_name=tensor_cache_element_lookup.get_tensor_cache_category_subset().get_name(),
							module_input=module_input,
							module_output=module_output,
							training_data_purpose_type=training_data_purpose_type
						)

						is_successful, tensor_cache_element_lookup = tensor_cache_category_subset_cycle.try_get_next_tensor_cache_element_lookup()
						index += 1

						if index % 1000 == 0:
							print(f"{datetime.utcnow()}: sent {index} training data")

			print(f"{datetime.utcnow()}: finished sending training data")
			server_messenger.stop_receiving_from_clients()

		finally:
			server_messenger.dispose()
			if client_structure is not None:
				client_structure.dispose()

	def test_yolov5_service_structure_initialize(self):

		server_messenger = get_default_server_messenger(
			service_structure_type=ServiceStructureTypeEnum.YoloV5ServiceStructure
		)

		server_messenger.start_receiving_from_clients()

		time.sleep(1.0)

		server_messenger.dispose()

	def test_yolov5_service_structure_send_training_data(self):

		used_directory_paths = [
			"./cache/git/yolov5/service_structure/staged/training/images",
			"./cache/git/yolov5/service_structure/staged/training/labels",
			"./cache/git/yolov5/service_structure/staged/validation/images",
			"./cache/git/yolov5/service_structure/staged/validation/labels",
			"./cache/git/yolov5/service_structure/active/training/images",
			"./cache/git/yolov5/service_structure/active/training/labels",
			"./cache/git/yolov5/service_structure/active/validation/images",
			"./cache/git/yolov5/service_structure/active/validation/labels"
		]
		for directory_path in used_directory_paths:
			if os.path.exists(directory_path):
				delete_directory_contents(
					directory_path=directory_path
				)

		random_instance = random.Random(0)
		words = get_random_words()
		random_instance.shuffle(words)

		images_total = 10

		server_messenger = get_default_server_messenger(
			service_structure_type=ServiceStructureTypeEnum.YoloV5ServiceStructure
		)

		try:

			time.sleep(1.0)

			client_structure = None  # type: ClientStructure

			try:
				self.assertIsNotNone(server_messenger)

				server_messenger.start_receiving_from_clients()

				time.sleep(0.5)

				client_structure = get_default_client_structure()

				image_fonts = [
					PIL.ImageFont.truetype("./resources/fonts/dark_academia/DarkAcademia-Regular.ttf", 20),
					PIL.ImageFont.truetype("./resources/fonts/dark_academia/DarkAcademia-Regular.ttf", 40)
				]

				image_size = (4032, 3024)

				for image_index, word in zip(range(images_total), words):

					image = PIL.Image.new("RGB", image_size, color=(0, 0, 0))
					image_font = random_instance.choice(image_fonts)  # type: PIL.ImageFont

					text_size = image_font.getsize(word)
					text_location = (
						random_instance.randrange(image_size[0] - text_size[0]),
						random_instance.randrange(image_size[1] - text_size[1])
					)
					text_color_float = get_random_rainbow_color(
						random_instance=random_instance
					)
					text_color = tuple([round(x * 255) for x in text_color_float])

					draw_image = PIL.ImageDraw.Draw(image)
					draw_image.text(text_location, word,
									fill=text_color,
									font=image_font)

					localizations = []
					localization = LocalizationModuleOutput(
						label_index=len(word) % 2,
						x=text_location[0] / image_size[0],
						y=text_location[1] / image_size[1],
						width=text_size[0] / image_size[0],
						height=text_size[1] / image_size[1]
					)
					localizations.append(localization)

					image_bytes = BytesIO()
					image_extension = "png"
					image.save(image_bytes, format=image_extension)
					image_bytes_base64string = base64.b64encode(image_bytes.getvalue()).decode()

					client_structure.add_training_data_announcement(
						category_set_name="word in image",
						category_subset_name=word,
						module_input=ImageModuleInput(
							image_bytes_base64string=image_bytes_base64string,
							image_extension=image_extension
						),
						module_output=LocalizationListModuleOutput.get_from_localization_list(
							localization_list=localizations
						),
						training_data_purpose_type=TrainingDataPurposeTypeEnum.Validation if (image_index + 1) % 10 == 0 else TrainingDataPurposeTypeEnum.Training
					)

					time.sleep(0.1)

				time.sleep(5)

			finally:
				if client_structure is not None:
					client_structure.dispose()

			server_messenger.stop_receiving_from_clients()

		finally:
			server_messenger.dispose()
