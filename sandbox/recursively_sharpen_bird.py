from __future__ import annotations
import os
import PIL.Image
import matplotlib.pyplot as plt
from src.austin_heller_repo.machine_learning.framework import Conv2dToLinearToConv2dModule, ModuleTrainer, TensorCacheCategorySubsetCycleRunner, ImageToImageModuleViewer
from src.austin_heller_repo.machine_learning.dataset.kaggle.gpiosenka_100_bird_species_sharpen_dataset import Gpiosenka100BirdSpeciesSharpenKaggleDataset


def main():

    is_cuda = True
    #tensor_cache_directory_path = "/media/austin/extradrive1/cache/tensor"
    #save_file_path = "/media/austin/extradrive1/cache/module/sharpen_birds.cm"
    tensor_cache_directory_path = "C:/cache/tensor"
    save_file_path = "C:/cache/module/sharpen_birds.cm"
    source_image_file_path = "C:/cache/temp/1.jpg"

    if os.path.exists(save_file_path):
        module_trainer = ModuleTrainer.load_from_file(
            module_class=Conv2dToLinearToConv2dModule,
            file_path=save_file_path,
            is_cuda=is_cuda
        )
    else:
        raise Exception(f"Failed to find existing module trainer at \"{save_file_path}\".")

    module = module_trainer.get_module()

    viewer = ImageToImageModuleViewer(
        module=module
    )

    next_image = PIL.Image.open(source_image_file_path)
    for _ in range(2):
        plt.imshow(next_image)
        plt.show()
        next_image = viewer.view_image(
            image=next_image
        )


if __name__ == '__main__':
    main()
