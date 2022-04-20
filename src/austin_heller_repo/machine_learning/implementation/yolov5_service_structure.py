from __future__ import annotations
import git
import os
import pathlib
import venv
import uuid
import shutil
import time
import tempfile
from typing import List, Dict, Tuple
from austin_heller_repo.common import delete_directory_contents, is_directory_empty, StringEnum, SubprocessWrapper
from austin_heller_repo.threading import Semaphore
from src.austin_heller_repo.machine_learning.service import ServiceStructure, StructureFactory, TrainingDataPurposeTypeEnum
from src.austin_heller_repo.machine_learning.common import ModuleInput, ModuleOutput, ImageModuleInput, LocalizationListModuleOutput, LocalizationModuleOutput


# TODO consider using: import logging


class YoloV5ModelTypeEnum(StringEnum):
	YoloV5N = "yolov5n"


class YoloV5ServiceStructure(ServiceStructure):

	def __init__(self, *, git_clone_directory_path: str, label_classes_total: int, is_cuda: bool, delay_between_training_seconds_total: float, is_git_pull_forced: bool, image_size: int, training_batch_size: int, model_type: YoloV5ModelTypeEnum, is_debug: bool = False):

		self.__git_clone_directory_path = git_clone_directory_path
		self.__label_classes_total = label_classes_total
		self.__is_cuda = is_cuda
		self.__is_git_pull_forced = is_git_pull_forced
		self.__image_size = image_size
		self.__training_batch_size = training_batch_size
		self.__model_type = model_type
		self.__is_debug = is_debug

		self.__yolov5_repo_directory_path = None  # type: str
		self.__yolov5_repo_venv_directory_path = None  # type: str
		self.__yolov5_repo_venv_activation_file_path = None  # type: str
		self.__yolov5_models_directory_path = None  # type: str
		self.__yolov5_training_model_file_name = None  # type: str
		self.__yolov5_training_model_file_path = None  # type: str
		self.__yolov5_detector_model_file_name = None  # type: str
		self.__yolov5_detector_model_file_path = None  # type: str
		self.__service_structure_directory_path = None  # type: str
		self.__staged_training_directory_path = None  # type: str
		self.__staged_training_image_directory_path = None  # type: str
		self.__staged_training_label_directory_path = None  # type: str
		self.__staged_validation_directory_path = None  # type: str
		self.__staged_validation_image_directory_path = None  # type: str
		self.__staged_validation_label_directory_path = None  # type: str
		self.__active_training_directory_path = None  # type: str
		self.__active_training_image_directory_path = None  # type: str
		self.__active_training_label_directory_path = None  # type: str
		self.__active_validation_directory_path = None  # type: str
		self.__active_validation_image_directory_path = None  # type: str
		self.__active_validation_label_directory_path = None  # type: str
		self.__service_yaml_file_name = None  # type: str
		self.__detector_model_semaphore = Semaphore()

		self.__initialize()

		super().__init__(
			delay_between_training_seconds_total=delay_between_training_seconds_total,
			is_debug=is_debug
		)

	def __initialize(self):

		self.__yolov5_repo_directory_path = os.path.join(self.__git_clone_directory_path, "yolov5")
		self.__yolov5_repo_venv_directory_path = os.path.join(self.__yolov5_repo_directory_path, "venv")
		self.__yolov5_models_directory_path = os.path.join(self.__yolov5_repo_directory_path, "models")
		self.__yolov5_training_model_file_name = "training.pt"
		self.__yolov5_training_model_file_path = os.path.join(self.__yolov5_models_directory_path, self.__yolov5_training_model_file_name)
		self.__yolov5_detector_model_file_name = "detector.pt"
		self.__yolov5_detector_model_file_path = os.path.join(self.__yolov5_models_directory_path, self.__yolov5_detector_model_file_name)
		self.__service_structure_directory_path = os.path.join(self.__yolov5_repo_directory_path, "service_structure")
		self.__staged_training_directory_path = os.path.join(self.__service_structure_directory_path, "staged", "training")
		self.__staged_training_image_directory_path = os.path.join(self.__staged_training_directory_path, "images")
		self.__staged_training_label_directory_path = os.path.join(self.__staged_training_directory_path, "labels")
		self.__staged_validation_directory_path = os.path.join(self.__service_structure_directory_path, "staged", "validation")
		self.__staged_validation_image_directory_path = os.path.join(self.__staged_validation_directory_path, "images")
		self.__staged_validation_label_directory_path = os.path.join(self.__staged_validation_directory_path, "labels")
		self.__active_training_directory_path = os.path.join(self.__service_structure_directory_path, "active", "training")
		self.__active_training_image_directory_path = os.path.join(self.__active_training_directory_path, "images")
		self.__active_training_label_directory_path = os.path.join(self.__active_training_directory_path, "labels")
		self.__active_validation_directory_path = os.path.join(self.__service_structure_directory_path, "active", "validation")
		self.__active_validation_image_directory_path = os.path.join(self.__active_validation_directory_path, "images")
		self.__active_validation_label_directory_path = os.path.join(self.__active_validation_directory_path, "labels")
		self.__service_yaml_file_name = "service_data.yaml"

		# setup yolov5 in git clone directory
		pathlib.Path(self.__yolov5_repo_directory_path).mkdir(parents=True, exist_ok=True)
		if is_directory_empty(
			directory_path=self.__yolov5_repo_directory_path
		):
			is_git_pull_required = True
		elif self.__is_git_pull_forced:
			delete_directory_contents(
				directory_path=self.__yolov5_repo_directory_path
			)
			is_git_pull_required = True
		else:
			is_git_pull_required = False

		if is_git_pull_required:
			git.Repo.clone_from("https://github.com/ultralytics/yolov5", self.__yolov5_repo_directory_path)
			venv.create(self.__yolov5_repo_venv_directory_path, with_pip=True)

			self.__yolov5_repo_venv_activation_file_path = self.__get_repo_venv_activation_file_path()

			if self.__is_cuda:
				torch_pip_install_command = "pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html"
			else:
				torch_pip_install_command = "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113"
			output_stream = os.popen(f"cd \"{self.__yolov5_repo_directory_path}\" && . \"{self.__yolov5_repo_venv_activation_file_path}\" && pip install -r requirements.txt && pip install albumentations wandb gsutil notebook rsa==4.7.2 && {torch_pip_install_command}")
			output = output_stream.read()
			output_stream.close()
			#print(f"output: {output}.")

			# setup training info xml file
			for directory_path in [
				self.__yolov5_models_directory_path,
				self.__staged_training_image_directory_path,
				self.__staged_training_label_directory_path,
				self.__staged_validation_image_directory_path,
				self.__staged_validation_label_directory_path,
				self.__active_training_image_directory_path,
				self.__active_training_label_directory_path,
				self.__active_validation_image_directory_path,
				self.__active_validation_label_directory_path
			]:
				pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)

			with open(os.path.join(self.__yolov5_repo_directory_path, "data", self.__service_yaml_file_name), "w") as file_handle:
				file_handle.writelines([
					f"train: {os.path.abspath(self.__active_training_directory_path)}\n",
					f"val: {os.path.abspath(self.__active_validation_directory_path)}\n",
					f"\n",
					f"# number of classes\n",
					f"nc: {self.__label_classes_total}\n",
					f"\n",
					f"# class names\n",
					f"names: [{','.join([f'class_{x}' for x in range(self.__label_classes_total)])}]"
				])
		else:  # the git repo already exists
			self.__yolov5_repo_venv_activation_file_path = self.__get_repo_venv_activation_file_path()

	def __get_repo_venv_activation_file_path(self) -> str:
		local_venv_activation_file_path_per_absolute_venv_activation_file_path = {
			os.path.join(self.__yolov5_repo_venv_directory_path, "bin", "activate"): "./venv/bin/activate",
			os.path.join(self.__yolov5_repo_venv_directory_path, "Scripts", "activate"): "./venv/Scripts/activate"
		}
		venv_activation_file_path = None
		for file_path in local_venv_activation_file_path_per_absolute_venv_activation_file_path:
			if os.path.exists(file_path):
				venv_activation_file_path = local_venv_activation_file_path_per_absolute_venv_activation_file_path[
					file_path]
				break
		if venv_activation_file_path is None:
			raise Exception(f"Failed to find venv activation in {self.__yolov5_repo_venv_directory_path}.")
		return venv_activation_file_path

	def __get_model_type_file_name(self) -> str:
		if self.__model_type == YoloV5ModelTypeEnum.YoloV5N:
			model_type_file_name = "yolov5n.yaml"
		else:
			raise Exception(f"Unexpected {YoloV5ModelTypeEnum.__name__} value: {self.__model_type.value}.")
		return model_type_file_name

	def get_module_output(self, *, module_input: ModuleInput) -> ModuleOutput:

		if not os.path.exists(self.__yolov5_detector_model_file_path):
			raise Exception(f"Failed to find detector model at \"{self.__yolov5_detector_model_file_path}\".")

		if isinstance(module_input, ImageModuleInput):
			image_bytes = module_input.get_image_bytes()
			image_extension = module_input.get_image_extension()
		else:
			raise Exception(f"Unexpected {ModuleInput.__name__} type: {type(module_input).__name__}.")

		image_uuid = str(uuid.uuid4())
		temp_directory = tempfile.TemporaryDirectory()

		try:
			image_file_path = os.path.join(temp_directory.name, f"{image_uuid}.{image_extension}")
			print(f"saving temp image to {image_file_path}")
			with open(image_file_path, "wb") as file_handle:
				file_handle.write(image_bytes)

			absolute_yolov5_repo_directory_path = os.path.abspath(self.__yolov5_repo_directory_path)

			detector_model_file_path = os.path.abspath(self.__yolov5_detector_model_file_path)

			os.environ['WANDB_SILENT'] = "true"  # stops the 30 second delay at the start

			command = f"cd \"{absolute_yolov5_repo_directory_path}\" ; . \"{self.__yolov5_repo_venv_activation_file_path}\" ; python detect.py --source \"{image_file_path}\" --img {self.__image_size} --weights \"{detector_model_file_path}\" --save-txt --nosave"

			subprocess_wrapper = SubprocessWrapper(
				command="sh",
				arguments=["-c", f"{command}"]
			)

			exit_code, output = subprocess_wrapper.run()

			print(f"exit_code: {exit_code}")
			print(f"output: {output}")

			localization_list_module_output = None  # type: LocalizationListModuleOutput

			if exit_code == 0:
				output_lines = output.split("\n")
				if len(output_lines) > 3 and output_lines[-3].startswith("Results saved to "):
					output_line = output_lines[-3]
					saved_detection_path_part = output_line[output_line.index("runs/detect/exp"):][:-4]

					saved_detection_directory_path = os.path.join(self.__yolov5_repo_directory_path, saved_detection_path_part)
					saved_detection_labels_directory_path = os.path.join(saved_detection_directory_path, "labels")

					if not os.path.exists(saved_detection_labels_directory_path):
						raise Exception(f"Failed to find detection labels path at: \"{saved_detection_labels_directory_path}\".")

					for file_name in os.listdir(saved_detection_labels_directory_path):
						file_path = os.path.join(saved_detection_labels_directory_path, file_name)
						if localization_list_module_output is not None:
							raise Exception(f"Unexpected additional file in output directory: \"{file_path}\".")
						localization_list_module_output = LocalizationListModuleOutput.get_from_localization_file(
							file_path=file_path
						)
					else:
						if localization_list_module_output is None:
							localization_list_module_output = LocalizationListModuleOutput.get_from_localization_list(
								 localization_list=[]
							)

			print(f"localization_list_module_output: {localization_list_module_output}")
			return localization_list_module_output

		finally:
			temp_directory.cleanup()

	def add_training_data(self, *, category_set_name: str, category_subset_name: str, module_input: ModuleInput, module_output: ModuleOutput, purpose: TrainingDataPurposeTypeEnum):

		if purpose == TrainingDataPurposeTypeEnum.Training:
			training_image_directory_path = self.__staged_training_image_directory_path
			training_label_directory_path = self.__staged_training_label_directory_path
		elif purpose == TrainingDataPurposeTypeEnum.Validation:
			training_image_directory_path = self.__staged_validation_image_directory_path
			training_label_directory_path = self.__staged_validation_label_directory_path
		else:
			raise Exception(f"Unexpected {TrainingDataPurposeTypeEnum.__name__} value: {purpose.value}")

		if isinstance(module_input, ImageModuleInput):
			image_uuid = str(uuid.uuid4())
			image_file_path = os.path.join(training_image_directory_path, f"{image_uuid}.{module_input.get_image_extension()}")
			annotation_file_path = os.path.join(training_label_directory_path, f"{image_uuid}.txt")
			image_bytes = module_input.get_image_bytes()
		else:
			raise Exception(f"Unexpected {ModuleInput.__name__} type: {type(module_input).__name__}.")

		if isinstance(module_output, LocalizationListModuleOutput):
			localizations = module_output.get_localizations()
		else:
			raise Exception(f"Unexpected {ModuleOutput.__name__} type: {type(module_output).__name__}.")

		try:
			with open(image_file_path, "wb") as file_handle:
				file_handle.write(image_bytes)
			with open(annotation_file_path, "w") as file_handle:
				for localization_index, localization in enumerate(localizations):
					if localization_index != 0:
						file_handle.write("\n")
					localization_string = f"{localization.get_label_index()} {localization.get_x()} {localization.get_y()} {localization.get_width()} {localization.get_height()}"
					file_handle.write(localization_string)
		except Exception:
			try:
				if os.path.exists(image_file_path):
					os.unlink(image_file_path)
			except Exception as _:
				pass

			try:
				if os.path.exists(annotation_file_path):
					os.unlink(annotation_file_path)
			except Exception as _:
				pass

			raise

	def train_module(self):

		# copy over staged training data into the active directory
		for source_image_directory_path, source_label_directory_path, destination_image_directory_path, destination_label_directory_path in [
			(self.__staged_training_image_directory_path, self.__staged_training_label_directory_path, self.__active_training_image_directory_path, self.__active_training_label_directory_path),
			(self.__staged_validation_image_directory_path, self.__staged_validation_label_directory_path, self.__active_validation_image_directory_path, self.__active_validation_label_directory_path)
		]:
			for image_file_name in os.listdir(source_image_directory_path):
				image_file_path = os.path.join(source_image_directory_path, image_file_name)
				print(f"found image: {image_file_path}")
				if os.path.isfile(image_file_path):
					image_uuid = os.path.splitext(image_file_name)[0]
					label_file_name = f"{image_uuid}.txt"
					label_file_path = os.path.join(source_label_directory_path, label_file_name)

					# construct destination file path details
					destination_image_file_path = os.path.join(destination_image_directory_path, image_file_name)
					destination_label_file_path = os.path.join(destination_label_directory_path, label_file_name)

					# if the data already exists, this process may have errored when it tried to remove the source files
					if os.path.exists(destination_image_file_path):
						os.unlink(destination_image_file_path)
					if os.path.exists(destination_label_file_path):
						os.unlink(destination_label_file_path)

					# copy source to destination
					try:
						shutil.copy(label_file_path, destination_label_file_path)
						shutil.copy(image_file_path, destination_image_file_path)
					except Exception:
						try:
							if os.path.exists(destination_label_file_path):
								os.unlink(destination_label_file_path)
						except Exception as _:
							pass
						try:
							if os.path.exists(destination_image_file_path):
								os.unlink(destination_image_file_path)
						except Exception as _:
							pass
						raise

					os.unlink(image_file_path)
					os.unlink(label_file_path)

		# start training module
		if not is_directory_empty(
			directory_path=self.__active_training_image_directory_path
		) and not is_directory_empty(
			directory_path=self.__active_validation_image_directory_path
		):
			model_type_file_name = self.__get_model_type_file_name()

			absolute_yolov5_repo_directory_path = os.path.abspath(self.__yolov5_repo_directory_path)

			if os.path.exists(self.__yolov5_training_model_file_path):
				training_model_file_path = os.path.abspath(self.__yolov5_training_model_file_path)
			else:
				training_model_file_path = ""

			os.environ['WANDB_SILENT'] = "true"  # stops the 30 second delay at the start

			command = f"cd \"{absolute_yolov5_repo_directory_path}\" ; . \"{self.__yolov5_repo_venv_activation_file_path}\" ; python train.py --img {self.__image_size} --cfg {model_type_file_name} --batch {self.__training_batch_size} --epochs 1 --data {self.__service_yaml_file_name} --weights \"{training_model_file_path}\""

			subprocess_wrapper = SubprocessWrapper(
				command="sh",
				arguments=["-c", f"{command}"]
			)

			exit_code, output = subprocess_wrapper.run()

			if exit_code == 0:
				output_lines = output.split("\n")
				if len(output_lines) > 2 and output_lines[-2].startswith("Results saved to "):
					output_line = output_lines[-2]
					saved_model_path_part = output_line[output_line.index("runs/train/exp"):][:-4]

					saved_model_directory_path = os.path.join(self.__yolov5_repo_directory_path, saved_model_path_part)
					saved_model_path = os.path.join(saved_model_directory_path, "weights", "last.pt")

					if not os.path.exists(saved_model_path):
						raise Exception(f"Failed to find model path at: \"{saved_model_path}\".")

					if os.path.exists(self.__yolov5_detector_model_file_path):
						os.unlink(self.__yolov5_detector_model_file_path)

					self.__detector_model_semaphore.acquire()
					shutil.copy(saved_model_path, self.__yolov5_detector_model_file_path)
					self.__detector_model_semaphore.release()

					if os.path.exists(self.__yolov5_training_model_file_path):
						os.unlink(self.__yolov5_training_model_file_path)

					shutil.copy(saved_model_path, self.__yolov5_training_model_file_path)

					shutil.rmtree(saved_model_directory_path)
				else:
					raise Exception(f"Failed to find \"Results saved\" line: {output_lines}")


class YoloV5ServiceStructureFactory(StructureFactory):

	def __init__(self, *, git_clone_directory_path: str, label_classes_total: int, is_cuda: bool, delay_between_training_seconds_total: float, is_git_pull_forced: bool, image_size: int, training_batch_size: int, model_type: YoloV5ModelTypeEnum, is_debug: bool = False):

		self.__git_clone_directory_path = git_clone_directory_path
		self.__label_classes_total = label_classes_total
		self.__is_cuda = is_cuda
		self.__delay_between_training_seconds_total = delay_between_training_seconds_total
		self.__is_git_pull_forced = is_git_pull_forced
		self.__image_size = image_size
		self.__training_batch_size = training_batch_size
		self.__model_type = model_type
		self.__is_debug = is_debug

	def get_structure(self) -> YoloV5ServiceStructure:
		return YoloV5ServiceStructure(
			git_clone_directory_path=self.__git_clone_directory_path,
			label_classes_total=self.__label_classes_total,
			is_cuda=self.__is_cuda,
			delay_between_training_seconds_total=self.__delay_between_training_seconds_total,
			is_git_pull_forced=self.__is_git_pull_forced,
			image_size=self.__image_size,
			training_batch_size=self.__training_batch_size,
			model_type=self.__model_type,
			is_debug=self.__is_debug
		)
