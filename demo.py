import torch
import numpy as np
import torch.nn as nn
import os, glob
from torchvision import transforms
from torchvision import models
from PIL import Image
from PFV import pfv
import matplotlib.pyplot as plt

# Embedding hook
embeddings = []
def feature_map_hook(module, input, output):
    embeddings.append(input[0])

def register_embedding_hooks_vgg(model):
    for i,module in enumerate(model.children()):
        # If the layer has children of its own, recursively traverse
        if list(module.children()): 
            register_embedding_hooks_vgg(module)
        # If a pooling layer, register a hook
        elif isinstance(module, nn.MaxPool2d):# or isinstance(module, nn.AdaptiveAvgPool2d):
            print(f"Registering hook to Layer ({i}) {module}")
            module.register_forward_hook(feature_map_hook)
        # If a strided convolution layer, register a hook
        elif isinstance(module, nn.Conv2d) and module.stride > (1,1):
                module.register_forward_hook(feature_map_hook)        
        elif isinstance(module, nn.Conv2d):
            if module.stride > (1,1):
                module.register_forward_hook(feature_map_hook)


if __name__ == '__main__':
    # Load images 
    image_dir = './sample_images/'
    image_files = glob.glob(image_dir + '*.jpg')
    input_images = []
    for f in image_files:
        input_images.append(Image.open(f))

    # Preprocess images
    input_image = input_images
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the input tensor
    input_tensor = []
    for image in input_images:
        input_tensor.append(preprocess(image))
    input_batch = torch.stack(input_tensor)

    # Initialize model
    model = models.vgg16(pretrained=True)
    # Register embedding hooks
    register_embedding_hooks_vgg(model)
    
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Run the batch through the network
    with torch.no_grad():
        output = model(input_batch)

    # Extract visualisation
    vis = pfv(embeddings, image_shape=input_batch.shape[-2:], idx_layer=len(embeddings)-1, hierarchical=True)


    def concat(imgs, f=lambda x: x):
        imgs = np.transpose(imgs, (0, 2, 3, 1))
        l = [f(imgs[i,:,:,:]) for i in range(imgs.shape[0])]
        return np.concatenate(l, axis=1)

    normalize = lambda x: (x - x.min()) / (x.max() - x.min())

    imgs = input_batch.detach().cpu().numpy()
    orig_imgs = concat(imgs, normalize)
    vis_imgs = concat(vis)

    # Make a mosaic of original and visualization images
    fig = np.concatenate([orig_imgs] + [vis_imgs])
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(fig)
    plt.savefig('result.png', bbox_inches='tight')
    plt.show()







    

