from __future__ import annotations
import os
from src.austin_heller_repo.machine_learning.framework import Conv2dToLinearToConv2dModule, ModuleTrainer, TensorCacheCategorySubsetCycleRunner
from src.austin_heller_repo.machine_learning.dataset.kaggle.gpiosenka_100_bird_species_sharpen_dataset import Gpiosenka100BirdSpeciesSharpenKaggleDataset


def main():

	is_cuda = True
	#tensor_cache_directory_path = "/media/austin/extradrive1/cache/tensor"
	#save_file_path = "/media/austin/extradrive1/cache/module/sharpen_birds.cm"
	tensor_cache_directory_path = "C:/cache/tensor"
	save_file_path = "C:/cache/module/sharpen_birds_10000.cm"

	if os.path.exists(save_file_path):
		module_trainer = ModuleTrainer.load_from_file(
			module_class=Conv2dToLinearToConv2dModule,
			file_path=save_file_path,
			is_cuda=is_cuda
		)
	else:
		module_trainer = ModuleTrainer(
			module=Conv2dToLinearToConv2dModule(
				input_size=(223, 223),
				input_channels=3,
				input_to_linear_layer_total=3,
				linear_layer_lengths=(100, 100),
				linear_to_output_layer_total=3,
				output_size=(223, 223),
				output_channels=3
			),
			is_categorical=False,
			is_recurrent=False,
			is_cuda=is_cuda
		)

	module_trainer.train(
		module_input=Gpiosenka100BirdSpeciesSharpenKaggleDataset().get_tensor_cache_category_subset_cycle(
			tensor_cache_directory_path=tensor_cache_directory_path
		),
		name="sharpen_birds",
		cache_length=100,
		learn_rate=0.01,
		epochs=1
	)

	module_trainer.save_to_file(
		file_path=save_file_path
	)


if __name__ == '__main__':
	main()
