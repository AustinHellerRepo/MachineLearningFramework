from __future__ import annotations
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod


class AbstractClass(ABC):

	def __init__(self, *, abc_arg: str):

		self.__abc_arg = abc_arg

	def get_abc_arg(self) -> str:
		return self.__abc_arg

	@classmethod
	def parse_json(cls, *, json_dict: Dict) -> AbstractClass:
		return cls(**json_dict)

	@abstractmethod
	def to_json(self) -> Dict:
		return {
			"abc_arg": self.__abc_arg
		}


class Implementation(AbstractClass):

	def __init__(self, *, test: str, abc_arg: str):
		super().__init__(
			abc_arg=abc_arg
		)

		self.__test = test

	def get_test(self) -> str:
		return self.__test

	def to_json(self) -> Dict:
		json_dict = super().to_json()
		json_dict["test"] = self.__test
		return json_dict


implementation = Implementation(
	test="first",
	abc_arg="second"
)

json_dict = implementation.to_json()

parsed_implementation = implementation.parse_json(
	json_dict=json_dict
)

print(f"implementation: {implementation.get_abc_arg()} {implementation.get_test()}")
print(f"parsed_implementation: {parsed_implementation.get_abc_arg()} {parsed_implementation.get_test()}")
