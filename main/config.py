import argparse

def str2bool(v):
  return v.lower() in ('true', '1', 'y', 'yes')

parser = argparse.ArgumentParser()
basic_arg = parser.add_argument_group("basic")
basic_arg.add_argument('--device', type=str, default="cpu:0")

train_arg = parser.add_argument_group("train")
train_arg.add_argument('--batch_size', type=int, default=4)
train_arg.add_argument('--max_iter', type=int, default=1000)
train_arg.add_argument('--mask', type=str2bool, default=False)
train_arg.add_argument('--ckpt_dir', type=str, default="ckpt")

data_arg = parser.add_argument_group("data")
data_arg.add_argument('--dataset', type=str, default="MSCOCO")
data_arg.add_argument('--data_dir', type=str, default="dataset")
data_arg.add_argument('--result_dir', type=str, default="result")
train_arg.add_argument('--mask_inv', type=str2bool, default=False)

def get_config():
  return parser.parse_args()
