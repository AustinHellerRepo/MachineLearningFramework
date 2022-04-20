from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Type
import torch
import base64
from austin_heller_repo.common import StringEnum, JsonParsable


class ModuleInputTypeEnum(StringEnum):
	Image = "image"
	FloatTensor = "float_tensor"


class ModuleInput(JsonParsable, ABC):

	def __init__(self):
		super().__init__()

		pass

	@classmethod
	def get_json_parsable_type_dictionary_key(cls) -> str:
		return "__module_input_type"


class ImageModuleInput(ModuleInput):

	def __init__(self, *, image_bytes_base64string: str, image_extension: str):
		super().__init__()

		self.__image_bytes_base64string = image_bytes_base64string
		self.__image_extension = image_extension

	def get_image_bytes_base64string(self) -> str:
		return self.__image_bytes_base64string

	def get_image_extension(self) -> str:
		return self.__image_extension

	def get_image_bytes(self) -> bytes:
		return base64.b64decode(self.__image_bytes_base64string.encode())

	@classmethod
	def get_json_parsable_type(cls) -> ModuleInputTypeEnum:
		return ModuleInputTypeEnum.Image

	def to_json(self) -> Dict:
		json_dict = super().to_json()
		json_dict["image_bytes_base64string"] = self.__image_bytes_base64string
		json_dict["image_extension"] = self.__image_extension
		return json_dict


class FloatTensorModuleInput(ModuleInput):

	def __init__(self, *, tensor_list: List):
		super().__init__()

		self.__tensor_list = tensor_list

	def get_float_tensor(self) -> torch.FloatTensor:
		return torch.FloatTensor(self.__tensor_list)

	@classmethod
	def get_json_parsable_type(cls) -> ModuleInputTypeEnum:
		return ModuleInputTypeEnum.FloatTensor

	def to_json(self) -> Dict:
		json_dict = super().to_json()
		json_dict["tensor_list"] = self.__tensor_list
		return json_dict

	@classmethod
	def get_from_float_tensor(cls, *, float_tensor: torch.FloatTensor) -> FloatTensorModuleInput:
		return FloatTensorModuleInput(
			tensor_list=float_tensor.tolist()
		)


class ModuleOutputTypeEnum(StringEnum):
	Text = "text"
	FloatTensor = "float_tensor"
	LongTensor = "long_tensor"
	Localization = "localization"
	LocalizationList = "localization_list"


class ModuleOutput(JsonParsable, ABC):

	def __init__(self):
		super().__init__()

		pass

	@classmethod
	def get_json_parsable_type_dictionary_key(cls) -> str:
		return "__module_output_type"


class TextModuleOutput(ModuleOutput):

	def __init__(self, *, text: str):
		super().__init__()

		self.__text = text

	def get_text(self) -> str:
		return self.__text

	@classmethod
	def get_json_parsable_type(cls) -> ModuleOutputTypeEnum:
		return ModuleOutputTypeEnum.Text

	def to_json(self) -> Dict:
		json_dict = super().to_json()
		json_dict["text"] = self.__text
		return json_dict


class FloatTensorModuleOutput(ModuleOutput):

	def __init__(self, *, tensor_list: List):
		super().__init__()

		self.__tensor_list = tensor_list

	def get_float_tensor(self) -> torch.FloatTensor:
		return torch.FloatTensor(self.__tensor_list)

	@classmethod
	def get_json_parsable_type(cls) -> ModuleOutputTypeEnum:
		return ModuleOutputTypeEnum.FloatTensor

	def to_json(self) -> Dict:
		json_dict = super().to_json()
		json_dict["tensor_list"] = self.__tensor_list
		return json_dict

	@classmethod
	def get_from_float_tensor(cls, *, float_tensor: torch.FloatTensor) -> FloatTensorModuleOutput:
		return FloatTensorModuleOutput(
			tensor_list=float_tensor.tolist()
		)


class LongTensorModuleOutput(ModuleOutput):

	def __init__(self, *, tensor_list: List):
		super().__init__()

		self.__tensor_list = tensor_list

	def get_long_tensor(self) -> torch.LongTensor:
		return torch.LongTensor(self.__tensor_list)

	@classmethod
	def get_json_parsable_type(cls) -> ModuleOutputTypeEnum:
		return ModuleOutputTypeEnum.LongTensor

	def to_json(self) -> Dict:
		json_dict = super().to_json()
		json_dict["tensor_list"] = self.__tensor_list
		return json_dict

	@classmethod
	def get_from_long_tensor(cls, *, long_tensor: torch.LongTensor) -> LongTensorModuleOutput:
		return LongTensorModuleOutput(
			tensor_list=long_tensor.tolist()
		)


class LocalizationModuleOutput(ModuleOutput):

	def __init__(self, *, label_index: int, x: float, y: float, width: float, height: float):
		super().__init__()

		self.__label_index = label_index
		self.__x = x
		self.__y = y
		self.__width = width
		self.__height = height

		self.__initialize()

	def __initialize(self):
		if self.__x < 0 or self.__x > 1:
			raise Exception(f"Localization x must be between 0 and 1.")
		if self.__y < 0 or self.__y > 1:
			raise Exception(f"Localization x must be between 0 and 1.")
		if self.__width < 0 or self.__width > 1:
			raise Exception(f"Localization x must be between 0 and 1.")
		if self.__height < 0 or self.__height > 1:
			raise Exception(f"Localization x must be between 0 and 1.")

	def get_label_index(self) -> int:
		return self.__label_index

	def get_x(self) -> float:
		return self.__x

	def get_y(self) -> float:
		return self.__y

	def get_width(self) -> float:
		return self.__width

	def get_height(self) -> float:
		return self.__height

	@classmethod
	def get_json_parsable_type(cls) -> ModuleOutputTypeEnum:
		return ModuleOutputTypeEnum.Localization

	def to_json(self) -> Dict:
		json_dict = super().to_json()
		json_dict["label_index"] = self.__label_index
		json_dict["x"] = self.__x
		json_dict["y"] = self.__y
		json_dict["width"] = self.__width
		json_dict["height"] = self.__height
		return json_dict


class LocalizationListModuleOutput(ModuleOutput):

	def __init__(self, *, localization_json_dicts: List[Dict]):
		super().__init__()

		self.__localization_json_dicts = localization_json_dicts

	def get_localizations(self) -> List[LocalizationModuleOutput]:
		return [LocalizationModuleOutput.parse_json(
			json_dict=x
		) for x in self.__localization_json_dicts]

	@classmethod
	def get_json_parsable_type(cls) -> ModuleOutputTypeEnum:
		return ModuleOutputTypeEnum.LocalizationList

	def to_json(self) -> Dict:
		json_dict = super().to_json()
		json_dict["localization_json_dicts"] = self.__localization_json_dicts
		return json_dict

	@classmethod
	def get_from_localization_list(cls, *, localization_list: List[LocalizationModuleOutput]) -> LocalizationListModuleOutput:
		return LocalizationListModuleOutput(
			localization_json_dicts=[x.to_json() for x in localization_list]
		)

	@staticmethod
	def get_from_localization_file(*, file_path: str) -> LocalizationListModuleOutput:
		localizations = []  # type: List[LocalizationModuleOutput]
		with open(file_path, "r") as file_handle:
			for line in file_handle.readlines():
				line_parts = line.split(" ")
				localization = LocalizationModuleOutput(
					label_index=int(line_parts[0]),
					x=float(line_parts[1]),
					y=float(line_parts[2]),
					width=float(line_parts[3]),
					height=float(line_parts[4])
				)
				localizations.append(localization)
		return LocalizationListModuleOutput.get_from_localization_list(
			localization_list=localizations
		)
