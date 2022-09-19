import torch
import torchvision.models as models
import torchvision.transforms as trs
import numpy as np
import PIL.Image as Image


def extract_features(img):

    # print(type(img[0, 0, 0]))

    img = Image.fromarray(np.uint8(img * 255))

    device = 'cuda:0'

    model_type = 'vgg'

    if model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    if model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_type == 'vgg':
        model = models.vgg16(pretrained=True).features
        avg_pool = torch.nn.AdaptiveAvgPool2d(1)

    model.to(device)
    model.eval()

    transform = trs.Compose(
        [
            trs.Resize([224, 224]),
            trs.ToTensor(),
            trs.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
        ]
    )

    inpt = transform(img)
    inpt = inpt.to(device)

    with torch.no_grad():
        feat = model(inpt.unsqueeze(0))

        if model_type == 'vgg':
            feat = avg_pool(feat)

    return feat.cpu().numpy().squeeze()


out = extract_features(image)
