from __future__ import annotations
from typing import List, Tuple, Dict, Type
from abc import ABC, abstractmethod
import json
from collections import deque
import uuid
import os
import shutil
from datetime import datetime
import inspect
import time
import torch
from austin_heller_repo.socket_queued_message_framework import SourceTypeEnum, ClientServerMessage, ClientServerMessageTypeEnum, StructureStateEnum, StructureTransitionException, Structure, StructureFactory, StructureInfluence, ClientMessengerFactory
from austin_heller_repo.threading import Semaphore, start_thread
from austin_heller_repo.common import StringEnum
from src.austin_heller_repo.machine_learning.common import ModuleInput, ModuleOutput


class TrainingDataPurposeTypeEnum(StringEnum):
	Training = "training"
	Validation = "validation"


class ServiceSourceTypeEnum(SourceTypeEnum):
	Service = "service"
	Client = "client"


class ServiceStructureStateEnum(StructureStateEnum):
	Active = "active"


class ClientStructureStateEnum(StructureStateEnum):
	Active = "active"


class ServiceClientServerMessageTypeEnum(ClientServerMessageTypeEnum):
	# service
	ServiceError = "service_error"
	# client
	DetectionRequest = "detection_request"
	DetectionResponse = "detection_response"
	AddTrainingDataAnnouncement = "add_training_data_announcement"


class ServiceClientServerMessage(ClientServerMessage, ABC):

	def __init__(self, *, destination_uuid: str):
		super().__init__(
			destination_uuid=destination_uuid
		)

		pass

	@classmethod
	def get_client_server_message_type_class(cls) -> Type[ClientServerMessageTypeEnum]:
		return ServiceClientServerMessageTypeEnum


###############################################################################
# Service
###############################################################################

class ServiceErrorServiceClientServerMessage(ServiceClientServerMessage):

	def __init__(self, *, structure_state_name: str, client_server_message_json_string: str, destination_uuid: str):
		super().__init__(
			destination_uuid=destination_uuid
		)

		self.__structure_state_name = structure_state_name
		self.__client_server_message_json_string = client_server_message_json_string

	def get_structure_state(self) -> ServiceStructureStateEnum:
		return ServiceStructureStateEnum(self.__structure_state_name)

	def get_client_server_message(self) -> ServiceClientServerMessage:
		return ServiceClientServerMessage.parse_from_json(
			json_object=json.loads(self.__client_server_message_json_string)
		)

	@classmethod
	def get_client_server_message_type(cls) -> ClientServerMessageTypeEnum:
		return ServiceClientServerMessageTypeEnum.ServiceError

	def to_json(self) -> Dict:
		json_object = super().to_json()
		json_object["structure_state_name"] = self.__structure_state_name
		json_object["client_server_message_json_string"] = self.__client_server_message_json_string
		return json_object

	def get_structural_error_client_server_message_response(self, *, structure_transition_exception: StructureTransitionException, destination_uuid: str) -> ClientServerMessage:
		return None


###############################################################################
# Client
###############################################################################

class DetectionRequestServiceClientServerMessage(ServiceClientServerMessage):

	def __init__(self, *, module_input_json_dict: Dict, destination_uuid: str):
		super().__init__(
			destination_uuid=destination_uuid
		)

		self.__module_input_json_dict = module_input_json_dict

	def get_module_input(self) -> ModuleInput:
		return ModuleInput.parse_json(
			json_dict=self.__module_input_json_dict
		)

	@classmethod
	def get_client_server_message_type(cls) -> ClientServerMessageTypeEnum:
		return ServiceClientServerMessageTypeEnum.DetectionRequest

	def to_json(self) -> Dict:
		json_object = super().to_json()
		json_object["module_input_json_dict"] = self.__module_input_json_dict
		return json_object

	def get_structural_error_client_server_message_response(self, *, structure_transition_exception: StructureTransitionException, destination_uuid: str) -> ClientServerMessage:
		return ServiceErrorServiceClientServerMessage(
			structure_state_name=structure_transition_exception.get_structure_state().value,
			client_server_message_json_string=json.dumps(structure_transition_exception.get_structure_influence().get_client_server_message().to_json()),
			destination_uuid=destination_uuid
		)


class DetectionResponseServiceClientServerMessage(ServiceClientServerMessage):

	def __init__(self, *, module_output_json_dict: Dict, destination_uuid: str):
		super().__init__(
			destination_uuid=destination_uuid
		)

		self.__module_output_json_dict = module_output_json_dict

	def get_module_output(self) -> ModuleOutput:
		return ModuleOutput.parse_json(
			json_dict=self.__module_output_json_dict
		)

	@classmethod
	def get_client_server_message_type(cls) -> ClientServerMessageTypeEnum:
		return ServiceClientServerMessageTypeEnum.DetectionResponse

	def to_json(self) -> Dict:
		json_object = super().to_json()
		json_object["module_output_json_dict"] = self.__module_output_json_dict
		return json_object

	def get_structural_error_client_server_message_response(self, *, structure_transition_exception: StructureTransitionException, destination_uuid: str) -> ClientServerMessage:
		return ServiceErrorServiceClientServerMessage(
			structure_state_name=structure_transition_exception.get_structure_state().value,
			client_server_message_json_string=json.dumps(structure_transition_exception.get_structure_influence().get_client_server_message().to_json()),
			destination_uuid=destination_uuid
		)


class AddTrainingDataAnnouncementServiceClientServerMessage(ServiceClientServerMessage):

	def __init__(self, *, category_set_name: str, category_subset_name: str, module_input_json_dict: Dict, module_output_json_dict: Dict, training_data_purpose_type_string: str, destination_uuid: str):
		super().__init__(
			destination_uuid=destination_uuid
		)

		self.__category_set_name = category_set_name
		self.__category_subset_name = category_subset_name
		self.__module_input_json_dict = module_input_json_dict
		self.__module_output_json_dict = module_output_json_dict
		self.__training_data_purpose_type_string = training_data_purpose_type_string

	def get_category_set_name(self) -> str:
		return self.__category_set_name

	def get_category_subset_name(self) -> str:
		return self.__category_subset_name

	def get_module_input(self) -> ModuleInput:
		return ModuleInput.parse_json(
			json_dict=self.__module_input_json_dict
		)

	def get_module_output(self) -> ModuleOutput:
		return ModuleOutput.parse_json(
			json_dict=self.__module_output_json_dict
		)

	def get_training_data_purpose_type(self) -> TrainingDataPurposeTypeEnum:
		return TrainingDataPurposeTypeEnum(self.__training_data_purpose_type_string)

	@classmethod
	def get_client_server_message_type(cls) -> ClientServerMessageTypeEnum:
		return ServiceClientServerMessageTypeEnum.AddTrainingDataAnnouncement

	def to_json(self) -> Dict:
		json_object = super().to_json()
		json_object["category_set_name"] = self.__category_set_name
		json_object["category_subset_name"] = self.__category_subset_name
		json_object["module_input_json_dict"] = self.__module_input_json_dict
		json_object["module_output_json_dict"] = self.__module_output_json_dict
		json_object["training_data_purpose_type_string"] = self.__training_data_purpose_type_string
		return json_object

	def get_structural_error_client_server_message_response(self, *, structure_transition_exception: StructureTransitionException, destination_uuid: str) -> ClientServerMessage:
		return ServiceErrorServiceClientServerMessage(
			structure_state_name=structure_transition_exception.get_structure_state().value,
			client_server_message_json_string=json.dumps(structure_transition_exception.get_structure_influence().get_client_server_message().to_json()),
			destination_uuid=destination_uuid
		)


class ClientStructure(Structure):

	def __init__(self, *, service_client_messenger_factory: ClientMessengerFactory):
		super().__init__(
			states=ClientStructureStateEnum,
			initial_state=ClientStructureStateEnum.Active
		)

		self.__service_client_messenger_factory = service_client_messenger_factory

		self.__service_source_uuid = None  # type: str
		self.__send_detection_request_blocking_semaphore = Semaphore()
		self.__send_detection_response_module_output = None  # type: ModuleOutput

		self.add_transition(
			client_server_message_type=ServiceClientServerMessageTypeEnum.DetectionResponse,
			from_source_type=ServiceSourceTypeEnum.Service,
			start_structure_state=ClientStructureStateEnum.Active,
			end_structure_state=ClientStructureStateEnum.Active,
			on_transition=self.__service_detection_response_transition
		)

		self.__initialize()

	def __initialize(self):

		self.__send_detection_request_blocking_semaphore.acquire()

		self.connect_to_outbound_messenger(
			client_messenger_factory=self.__service_client_messenger_factory,
			source_type=ServiceSourceTypeEnum.Service,
			tag_json=None
		)

	def __service_detection_response_transition(self, structure_influence: StructureInfluence):
		client_server_message = structure_influence.get_client_server_message()
		if isinstance(client_server_message, DetectionResponseServiceClientServerMessage):
			self.__send_detection_response_module_output = client_server_message.get_module_output()
			self.__send_detection_request_blocking_semaphore.release()
		else:
			raise Exception(f"Unexpected client server message type: {client_server_message}.")

	def on_client_connected(self, *, source_uuid: str, source_type: SourceTypeEnum, tag_json: Dict or None):
		if source_type == ServiceSourceTypeEnum.Service:
			self.__service_source_uuid = source_uuid
		else:
			raise Exception(f"Unexpected connection from source: {source_type.value}")

	def send_detection_request(self, *, module_input: ModuleInput) -> ModuleOutput:
		self.send_client_server_message(
			client_server_message=DetectionRequestServiceClientServerMessage(
				module_input_json_dict=module_input.to_json(),
				destination_uuid=self.__service_source_uuid
			)
		)
		self.__send_detection_request_blocking_semaphore.acquire()
		return self.__send_detection_response_module_output

	def add_training_data_announcement(self, *, category_set_name: str, category_subset_name: str, module_input: ModuleInput, module_output: ModuleOutput, training_data_purpose_type: TrainingDataPurposeTypeEnum):
		self.send_client_server_message(
			client_server_message=AddTrainingDataAnnouncementServiceClientServerMessage(
				category_set_name=category_set_name,
				category_subset_name=category_subset_name,
				module_input_json_dict=module_input.to_json(),
				module_output_json_dict=module_output.to_json(),
				training_data_purpose_type_string=training_data_purpose_type.value,
				destination_uuid=self.__service_source_uuid
			)
		)


class ServiceStructure(Structure, ABC):

	def __init__(self, *, delay_between_training_seconds_total: float, is_debug: bool = False):
		super().__init__(
			states=ServiceStructureStateEnum,
			initial_state=ServiceStructureStateEnum.Active
		)

		self.__delay_between_training_seconds_total = delay_between_training_seconds_total
		self.__is_debug = is_debug

		self.__is_training_module_thread_active = False

		self.add_transition(
			client_server_message_type=ServiceClientServerMessageTypeEnum.DetectionRequest,
			from_source_type=ServiceSourceTypeEnum.Client,
			start_structure_state=ServiceStructureStateEnum.Active,
			end_structure_state=ServiceStructureStateEnum.Active,
			on_transition=self.__client_detection_request_transition
		)

		self.add_transition(
			client_server_message_type=ServiceClientServerMessageTypeEnum.AddTrainingDataAnnouncement,
			from_source_type=ServiceSourceTypeEnum.Client,
			start_structure_state=ServiceStructureStateEnum.Active,
			end_structure_state=ServiceStructureStateEnum.Active,
			on_transition=self.__client_add_training_data_announcement_transition
		)

		self.__initialize()

	def __initialize(self):

		self.__training_module_thread = start_thread(self.__training_module_thread_method)

	def __client_detection_request_transition(self, structure_influence: StructureInfluence):

		client_server_message = structure_influence.get_client_server_message()
		if not isinstance(client_server_message, DetectionRequestServiceClientServerMessage):
			raise Exception(f"Unexpected message type: {client_server_message.__class__.get_client_server_message_type()}")
		else:
			try:
				module_output = self.get_module_output(
					module_input=client_server_message.get_module_input()
				)
				self.send_client_server_message(
					client_server_message=DetectionResponseServiceClientServerMessage(
						module_output_json_dict=module_output.to_json(),
						destination_uuid=structure_influence.get_source_uuid()
					)
				)
				if self.__is_debug:
					print(f"{datetime.utcnow()}: success: detected output for user {structure_influence.get_source_uuid()}.")
			except Exception as ex:
				if self.__is_debug:
					print(f"{datetime.utcnow()}: ex: {ex}.")
				raise

	def __client_add_training_data_announcement_transition(self, structure_influence: StructureInfluence):

		client_server_message = structure_influence.get_client_server_message()
		if not isinstance(client_server_message, AddTrainingDataAnnouncementServiceClientServerMessage):
			raise Exception(f"Unexpected message type: {client_server_message.__class__.get_client_server_message_type()}")
		else:
			try:
				self.add_training_data(
					category_set_name=client_server_message.get_category_set_name(),
					category_subset_name=client_server_message.get_category_subset_name(),
					module_input=client_server_message.get_module_input(),
					module_output=client_server_message.get_module_output(),
					purpose=client_server_message.get_training_data_purpose_type()
				)
				if self.__is_debug:
					print(f"{datetime.utcnow()}: success: added training data from {structure_influence.get_source_uuid()} for purpose {client_server_message.get_training_data_purpose_type().value}.")
			except Exception as ex:
				if self.__is_debug:
					print(f"{datetime.utcnow()}: ex: {ex}.")
				raise

	def on_client_connected(self, *, source_uuid: str, source_type: SourceTypeEnum, tag_json: Dict or None):
		if source_type == ServiceSourceTypeEnum.Client:
			pass
		else:
			raise Exception(f"Unexpected connection from source: {source_type.value}")

	@abstractmethod
	def get_module_output(self, *, module_input: ModuleInput) -> ModuleOutput:
		raise NotImplementedError()

	@abstractmethod
	def add_training_data(self, *, category_set_name: str, category_subset_name: str, module_input: ModuleInput, module_output: ModuleOutput, purpose: TrainingDataPurposeTypeEnum):
		raise NotImplementedError()

	@abstractmethod
	def train_module(self):
		raise NotImplementedError()

	def __training_module_thread_method(self):

		self.__is_training_module_thread_active = True
		try:
			while self.__is_training_module_thread_active:
				self.train_module()
				time.sleep(self.__delay_between_training_seconds_total)
		except Exception as ex:
			print(f"{datetime.utcnow()}: {inspect.stack()[0][3]}: ex: {ex}")
			raise

	def dispose(self):
		try:
			self.__is_training_module_thread_active = False
		finally:
			super().dispose()
