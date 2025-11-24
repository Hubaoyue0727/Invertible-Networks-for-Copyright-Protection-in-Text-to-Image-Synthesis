from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
from args import get_args_parser
from util.utils import *
import config as c
args = get_args_parser()

class Dataset(Dataset):
    def __init__(self, inputpath,transforms_=None,mode="train"):

        self.transform = transforms_
        self.TRAIN_PATH = inputpath
        # self.TEST_PATH = args.testpath
        self.format_train = 'png'
        # self.format_train = 'jpg'
        # self.format_train = 'JPEG'
        
        self.format_test = 'png'
        if mode == 'train':
            self.files = natsorted(sorted(imglist(self.TRAIN_PATH, self.format_train)))
        # assert len(self.files) > 0, f"No images found in {self.TRAIN_PATH} with format {self.format_train}"
        # if mode =='test':
        #     # test
        #     self.files = natsorted(sorted(imglist(self.TEST_PATH , self.format_test)))
        # assert len(self.files) > 0, f"No images found in {self.TEST_PATH} with format {self.format_test}"

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            # print(f"Loaded image at index {index}")
            image = to_rgb(image)
            item = self.transform(image)
            # item = item.unsqueeze(0)
            filename = self.files[index].split("/")
            classname = filename[len(filename)-3]  # class name is the third!!! last element in the path
            return item, classname

        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            return None, None

    def __len__(self):
        return len(self.files)

transform = T.Compose([
    T.Resize(c.imagesize, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(c.imagesize),
    T.ToTensor(),
])

# # Training data loader
# trainloader = DataLoader(
#     Dataset(transforms_=transform, mode="train"),
#     batch_size=1,
#     shuffle=False,
#     pin_memory=False,
#     num_workers=args.workers,
#     drop_last=True
# )

'''
# Test data loader
testloader = DataLoader(
    Dataset(transforms_=transform, mode="test"),
    batch_size=1,
    shuffle=False,
    pin_memory=False,
    num_workers=args.workers,
    drop_last=True
)
'''