import numpy as np
import pandas as pd
import PIL.Image as Image
import scipy.io as sio
import torch
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as trs

device = 'cuda:0'

model_type = 'vgg16'


class dataset(data.Dataset):

    def __init__(self):
        super(dataset, self).__init__()

        self.ds = '/home/zino/Datasets/a5k/img_all_adjusted_gamma_applied'
        self.annot = pd.read_csv('~/Datasets/a5k/train_cleaned_increased_2.csv')

        self.transform = trs.Compose(
            [
                trs.Resize([224, 224]),
                trs.ToTensor(),
                trs.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):

        img = Image.open(self.ds + '/' + str(self.annot.id[index]) + '.png')
        inpt = self.transform(img)
        name = str(self.annot.id[index])
        return inpt, name

    def __len__(self):
        return len(self.annot)



if __name__ == "__main__":
    if model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    if model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_type == 'vgg16':
        model = models.vgg16(pretrained=True).features
        avg_pool = torch.nn.AdaptiveAvgPool2d(1)

    model.to(device)
    model.eval()

    # save_dir = './features/features_' + model_type + '/'

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)


    data_loader = data.DataLoader(
        dataset=dataset(),
        num_workers=8,
        batch_size=128,
        shuffle=False,
        pin_memory=True
    )


    tt = trs.ToTensor()

    X_net = []
    Names = []

    print('Feature extraction:')


    for i, btch in enumerate(data_loader, 1):

        inpt, names = btch[0], btch[1]

        print('\r\r\r', end='\r')
        print('{}/{}'.format(i, len(data_loader)), end='\r')

        # print('{}.png'.format(annot.id[ii]), end='\r')

        inpt = inpt.to(device)

        with torch.no_grad():
            feat = model(inpt)

            if model_type == 'vgg16':
                feat = avg_pool(feat)
            if model_type == 'resnet18':
                feat = feat.squeeze()

        X_net.append(feat.cpu().numpy().squeeze())
        Names.append(names)


    X_net = np.concatenate(X_net)
    Names = np.concatenate(Names)

    # if model_type == 'resnet50':

    #     X_net = torch.pca_lowrank(tt(X_net), 48)
    #     X_net = X_net[0].squeeze().cpu().numpy()

    sio.savemat('./features/features_' + model_type +
                '_train_cleaned_increased.mat', {'X_net': X_net, 'filelist': Names})

    import ipdb

    ipdb.set_trace()
