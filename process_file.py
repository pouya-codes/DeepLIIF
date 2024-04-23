"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from deepliif.options.test_options import TestOptions
from deepliif.options import read_model_params, Options, print_options
from deepliif.data import create_dataset
from deepliif.models import create_model
from deepliif.util.visualizer import save_images
from deepliif.util import html
import torch
import click
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

class ImageProcessor:
    def __init__(self, checkpoint_dir='./checkpoints', model_name='name'):
        # hard-code some parameters for test
        self.opt = TestOptions().parse()  # get test options
        self.opt.checkpoints_dir = '/home/pouya/Develop/UBC/QA-QC/Codes/Models/DeepLIIF_Latest_Model'  # load models from here
        self.opt.name = "latest_net"
        self.opt.num_threads = 0   # test code only supports num_threads = 1
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        self.model = create_model(self.opt)      # create a model given opt.model and other options
        self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers

    def process_image(self, pil_image):
        # Convert PIL image to numpy array
        # input_image = np.array(pil_image)
        to_tensor = ToTensor()
        input_image = to_tensor(pil_image).unsqueeze(0)  # Add batch dimension

        # Set model input
        self.model.set_input({'A': input_image, 'A_paths': ''})  # 'A' and 'A_paths' are placeholders

        # Run model
        self.model.test()  # run forward pass

        # Get output
        visuals = self.model.get_current_visuals()  # get image results
        output_image_tensor = visuals['fake_B']  # 'fake_B' is the output image

        # Convert tensor to PIL image
        to_pil_image = ToPILImage()
        output_image = to_pil_image(output_image_tensor.squeeze(0))  # Remove batch dimension

        return output_image
    
if __name__ == "__main__":
    # Initialize image processor
    image_processor = ImageProcessor()

    # Load image
    image_path = "/home/pouya/Develop/UBC/QA-QC/Datasets/temp/braf 207_1_region.png"
    pil_image = Image.open(image_path).convert('RGB')
    print("ss")
    # Process image
    output_image = image_processor.process_image(pil_image)

    # Save output image
    output_image.save("path/to/output.jpg")
