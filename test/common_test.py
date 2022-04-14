from __future__ import annotations
import unittest
import torch
import numpy as np
import json
from src.austin_heller_repo.machine_learning.common import FloatTensorModuleInput


class CommonTest(unittest.TestCase):

	def test_convert_float_tensor_to_float_tensor_module_input_and_back(self):

		float_tensors = [
			torch.FloatTensor([0.1, 0.2, 0.3]),
			torch.rand(2, 3),
			torch.FloatTensor(np.random.random((4, 5, 3)))
		]

		for float_tensor in float_tensors:

			print(f"float_tensor: {float_tensor}")

			float_tensor_module_input = FloatTensorModuleInput.get_from_float_tensor(
				float_tensor=float_tensor
			)

			float_tensor_module_input_json_dict = float_tensor_module_input.to_json()

			float_tensor_module_input_json_string = json.dumps(float_tensor_module_input_json_dict)

			actual_float_tensor_module_input_json_dict = json.loads(float_tensor_module_input_json_string)

			actual_float_tensor_module_input = FloatTensorModuleInput.parse_json(
				json_dict=actual_float_tensor_module_input_json_dict
			)

			actual_float_tensor = actual_float_tensor_module_input.get_float_tensor()

			self.assertEqual(float_tensor.shape, actual_float_tensor.shape)
