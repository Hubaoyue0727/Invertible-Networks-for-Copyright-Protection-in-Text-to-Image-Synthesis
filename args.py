import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--inputpath_all', help='Path to the original image dataset to be protected.', default=r'')
    
    parser.add_argument('--copyrightpath', help='watermark image', default=r'')
    parser.add_argument('--outputpath', help='Directory to save checkpoints, logs, protected images, and so on.', default=r'./output')
    parser.add_argument('--pretrain', help='Use pretrained INN or not', default=False)
    
    parser.add_argument('--T2Imodel', help='Path to Text-to-image model', default="")

    # ------ For Recover ------
    parser.add_argument('--testmodelpath', help='Path to the trained model used for recovering', default=r'./output/2025-01-03_11-56-28/model/')
    parser.add_argument('--test_i', help='Test sample index', default='0')
    parser.add_argument('--test_img', help='Image ID', default='n000225')
    parser.add_argument('--adv_path', help='Path to adv images (copyrighted images) output by the inversible network', default='')
    parser.add_argument('--r_path', help='Path to restored images output by the inversible network', default='')
    parser.add_argument('--cover_path', help='Path to cover images (original images)', default='')    
    
    args = parser.parse_args()
    return args