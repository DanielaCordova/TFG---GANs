from torchvision.transforms import ToTensor, Normalize, Compose, Resize, RandomHorizontalFlip

def get_transform(difSize=None):
    if difSize is not None:
        imgTranformed = Compose([RandomHorizontalFlip(),Resize(difSize),ToTensor(),Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    else:
        imgTranformed = Compose([ RandomHorizontalFlip(), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return imgTranformed
