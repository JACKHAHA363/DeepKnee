"""
Demonstrate how to use inference
"""
import deepknee
from argparse import ArgumentParser


def get_args():
    """ Return arguments """
    parser = ArgumentParser()
    parser.add_argument('-img', required=True, type=str,
                        help='Path to input image')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    probs_df, cam, orig = deepknee.get_result(args.img)
    print(probs_df)
    orig.show()
    cam.show()
