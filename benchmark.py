dependencies = ['torch', 'torchvision']

from models.gen_efficientnet import tf_efficientnet_b0, tf_efficientnet_b1, tf_efficientnet_b2, tf_efficientnet_b3
from sotabench.image_classification import imagenet

def benchmark():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
        transforms.Resize(256, 'PIL.Image.BICUBIC'),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    imagenet.benchmark(
        model=tf_efficientnet_b0(num_classes=1000, pretrained=True, input_transform=input_transform),
        paper_model_name='EfficientNet B0',
        paper_arxiv_id='1905.11946',
        paper_pwc_id='efficientnet-rethinking-model-scaling-for'
    )
    
    input_transform = transforms.Compose([
        transforms.Resize(272, 'PIL.Image.BICUBIC'),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        normalize,
    ])
    
    imagenet.benchmark(
        model=tf_efficientnet_b1(num_classes=1000, pretrained=True, input_transform=input_transform),
        paper_model_name='EfficientNet B1',
        paper_arxiv_id='1905.11946',
        paper_pwc_id='efficientnet-rethinking-model-scaling-for'
    )
    
    input_transform = transforms.Compose([
        transforms.Resize(292, 'PIL.Image.BICUBIC'),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        normalize,
    ])
    
    imagenet.benchmark(
        model=tf_efficientnet_b2(num_classes=1000, pretrained=True, input_transform=input_transform),
        paper_model_name='EfficientNet B2',
        paper_arxiv_id='1905.11946',
        paper_pwc_id='efficientnet-rethinking-model-scaling-for'
    )
    
    input_transform = transforms.Compose([
        transforms.Resize(331, 'PIL.Image.BICUBIC'),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        normalize,
    ])
    
    imagenet.benchmark(
        model=tf_efficientnet_b3(num_classes=1000, pretrained=True, input_transform=input_transform),
        paper_model_name='EfficientNet B3',
        paper_arxiv_id='1905.11946',
        paper_pwc_id='efficientnet-rethinking-model-scaling-for'
    )
