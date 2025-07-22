import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='./Result')

# LPBA40
parser.add_argument("--atlas_file", type=str, help="gpu id number",
                    dest="atlas_file", default='../Datasets/LPBA40_delineation/delineation_l_norm/fixed.nii.gz')
# OASIS
# parser.add_argument("--atlas_file", type=str, help="gpu id number",
#                     dest="atlas_file", default='../Datasets/OASIS/fixed.nii.gz')
# IXI
# parser.add_argument("--atlas_file", type=str, help="gpu id number",
#                     dest="atlas_file", default='../Datasets/IXI_data/atlas.pkl')


# LPBA40 
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="../Datasets/LPBA40_delineation/delineation_l_norm/train")
# OASIS 
# parser.add_argument("--train_dir", type=str, help="data folder with training vols",
#                     dest="train_dir", default="../Datasets/OASIS/Train")
# IXI 
# parser.add_argument("--train_dir", type=str, help="data folder with training vols",
#                     dest="train_dir", default="../Datasets/IXI_data/Train")


parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=4e-4)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='ncc')
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=1)  
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='./Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='./Log')
# train
parser.add_argument('--epochs', default=2, type=int, help='epochs')
parser.add_argument('--save_model_dir', default='Checkpoint/', type=str, help='save model path')


# test
# LPBA40 
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='../Datasets/LPBA40_delineation/delineation_l_norm/test')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='../Datasets/LPBA40_delineation/label')
# OASIS 
# parser.add_argument("--test_dir", type=str, help="test data directory",
#                     dest="test_dir", default='../Datasets/OASIS/Test')
# parser.add_argument("--label_dir", type=str, help="label data directory",
#                     dest="label_dir", default='../Datasets/OASIS/label')
# OASIS 
# parser.add_argument("--test_dir", type=str, help="test data directory",
#                     dest="test_dir", default='../Datasets/IXI_data/Test')


args = parser.parse_args()
