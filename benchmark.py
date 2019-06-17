dependencies = ['scipy', 'torch', 'torchvision']

from models.gen_efficientnet import tf_efficientnet_b0, tf_efficientnet_b1, tf_efficientnet_b2, tf_efficientnet_b3
from sotabench.image_classification import imagenet

import torchvision.transforms as transforms
import PIL


def benchmark():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
        transforms.Resize(256, 'PIL.Image.BICUBIC'),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    imagenet.benchmark(
        model=tf_efficientnet_b0(num_classes=1000, pretrained=True),
        paper_model_name='EfficientNet',
        paper_arxiv_id='1905.11946',
        paper_pwc_id='efficientnet-rethinking-model-scaling-for',
        input_transform=input_transform,
        batch_size=256,
        num_gpu=1
    )
