import argparse

from pytorch_training import train
import logging_utils

LOGGER = logging_utils.initialize_logger(True)

class SmartFormatter(argparse.HelpFormatter):
  """Add option to split lines in help messages"""

  def _split_lines(self, text, width):
    # this is the RawTextHelpFormatter._split_lines
    if text.startswith('R|'):
        return text[2:].splitlines()
    return argparse.HelpFormatter._split_lines(self, text, width)

# class RemainderAction(argparse.Action):
#     """Action to create script attribute"""
#     def __call__(self, parser, namespace, values, option_string=None):
#         if not values:
#             raise argparse.ArgumentError(
#                 self, "can't be empty")

#         setattr(namespace, self.dest, values[0])
#         setattr(namespace, "argv", values) 

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=SmartFormatter)
  parser.add_argument("-v", "--version", action="version",
                      version="pytorch_client {}".format("1"))
  parser.add_argument("--dataset", action='store', type=str, help="Dataset to use, choosing from mnist, imagenet or cifar.", default="cifar")
  parser.add_argument("--ready", action='store_true', help="Set if data is loaded to the data source")
  parser.add_argument("--cpu", action='store_true', help="Using cpu for training.")
  parser.add_argument("--disk_source", action='store', type=str, help="Load dataset from disk and specifiy the path.", default="")
  parser.add_argument("--s3_source", action='store', type=str, help="Load dataset from S3 and specifiy the bucket.", default="cifar10-infinicache")
  parser.add_argument("--s3_train", action='store', type=str, help="Load training set from S3 and specifiy the bucket.", default="tianium.cifar10.training")
  parser.add_argument("--s3_test", action='store', type=str, help="Load test set from S3 and specifiy the bucket.", default="tianium.cifar10.test")
  parser.add_argument("--loader", action='store', type=str, help="Dataloader used, choosing from disk, s3, or infinicache.", default="")
  parser.add_argument("--model", action='store', type=str, help="Pretrained model, choosing from resnet, efficientnet or densenet.", default="")
  parser.add_argument("--batch", action='store', type=int, help="The size of batch.", default=64)
  parser.add_argument("--minibatch", action='store', type=int, help="The size of minibatch.", default=16)
  parser.add_argument("--epochs", action='store', type=int, help="Max epochs.", default=50)
  parser.add_argument("--accuracy", action='store', type=float, help="Target accuracy.", default=1.0)
  parser.add_argument("--benchmark", action='store_true', help="Simply benchmark the pretrained model.")
  parser.add_argument("--workers", action='store', type=int, help="Number of workers.", default=0)
  parser.add_argument("-o", "--output", action='store', type=str, help="Output file", default="")
  parser.add_argument("--prefix", action='store', type=str, help="Output prefix", default="")
  parser.add_argument("--test_mode", action='store_true', help="Test only")

  args, _ = parser.parse_known_args()
  train(args.__dict__)

main()