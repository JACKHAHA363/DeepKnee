""" Deepknee Inference API
    Author: Yuchen Lu
"""
from collections import OrderedDict
import glob
import os
import logging
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from pandas import DataFrame
import cv2

from deepknee.own_codes.model import KneeNet
from deepknee.own_codes.dataset import get_pair
from deepknee.own_codes.produce_gradcam import inverse_pair_mapping

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger('deep_knee')
LOGGER.setLevel(logging.INFO)
LOGGER.info('Version of pytorch: {}'.format(torch.__version__))

__all__ = ['get_result']


def _load_model(filename, net):
    LOGGER.info('Loading {} into mem...'.format(os.path.basename(filename)))
    state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
    try:
        net.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    net.eval()
    return net


# Args
DEEPKNEE_FOLDER = os.path.dirname(__file__)
SNAPSHOTS_FOLDER = os.path.join(DEEPKNEE_FOLDER, 'snapshots_knee_grading')
BW = 64

# Transform
MEAN, STD = np.load(os.path.join(SNAPSHOTS_FOLDER, 'mean_std.npy'))

# Loading models
LOGGER.info('loading models...')
MODELS = [_load_model(snp_name, KneeNet(BW, 0.2, True))
          for snp_name in glob.glob(os.path.join(SNAPSHOTS_FOLDER, '*', '*.pth'))]
LOGGER.info('done!')


class _KneeNetEnsemble(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                 lambda x: x.float(),
                                                 transforms.Normalize(MEAN, STD)])

        LOGGER.info('loading models...')
        LOGGER.info('loading models...')
        self.nets = [_load_model(snp_name, KneeNet(BW, 0.2, True))
                     for snp_name in glob.glob(os.path.join(SNAPSHOTS_FOLDER, '*', '*.pth'))]
        LOGGER.info('Done loading {} models...'.format(len(self.nets)))

        # Use only one branch
        for i, net in enumerate(self.nets):
            net.final = net.final[1]
            self.nets[i] = net
        self.grad_ls = []
        self.grad_ms = []
        self.softmax = torch.nn.Softmax(1)

    def load_img(self, fname):
        """ Load the image into (orig_img, lateral, medial) """
        LOGGER.info('Reading image {}...'.format(fname))
        orig_img = Image.open(fname)
        lateral, medial = get_pair(orig_img)
        return orig_img, self.img_transform(lateral), self.img_transform(medial)

    def compute_gradcam(self, l, m, img_size):
        """ Get heatmap """
        l_out = 0
        m_out = 0
        for net, wl, wm in zip(self.nets, self.grad_ls, self.grad_ms):
            ol, om = self.extract_features_branch(net, l, m, wl.data, wm.data)
            l_out += ol
            m_out += om
        l_out /= len(self.nets)
        m_out /= len(self.nets)

        heatmap = inverse_pair_mapping(l_out.data.cpu().numpy(),
                                       np.fliplr(m_out.data.cpu().numpy()),
                                       img_size)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        return heatmap

    def forward(self, l, m):
        """ Forwarding and register gradients """
        self.grad_ls = []
        self.grad_ms = []

        # Producing the branch outputs and registering the corresponding hooks
        # for attention maps
        # Ensemble by summing logits
        logits = 0
        for net in self.nets:
            l_o, m_o, o = self.decompose_forward_avg(net, l, m)
            l_o.register_hook(lambda grad: self.grad_ls.append(grad))
            m_o.register_hook(lambda grad: self.grad_ms.append(grad))
            logits += o
        return logits

    def predict(self, image_path):
        """Makes a prediction from file or a pre-loaded image
        :param image_path: Path to image
        :return: tuple
            image: A `Image`
            heatmap: A np array from 0 to 1
            probs: A np array
        """
        img, l, m = self.load_img(fname=image_path)
        self.train(True)
        self.zero_grad()
        logits = self.forward(Variable(l.unsqueeze(0)), Variable(m.unsqueeze(0)))
        probs = self.softmax(logits).data.cpu().numpy()

        # Backward use highest class
        pred_cls = torch.max(logits, dim=1)[1]
        oh_label = torch.FloatTensor(1, logits.size(1))
        oh_label.zero_()
        oh_label.scatter_(1, pred_cls.data.unsqueeze(0), 1)
        logits.backward(oh_label)
        heatmap = self.compute_gradcam(Variable(l.unsqueeze(0)),
                                       Variable(m.unsqueeze(0)),
                                       img_size=img.size)
        return img, heatmap, probs

    @staticmethod
    def decompose_forward_avg(net, l, m):
        l_o = net.branch(l)
        m_o = net.branch(m)

        concat = torch.cat([l_o, m_o], 1)
        o = net.final(concat.view(l.size(0), net.final.in_features))
        return l_o, m_o, o

    @staticmethod
    def extract_features_branch(net, l, m, wl, wm):
        def weigh_maps(weights, maps):
            maps = Variable(maps.squeeze())
            weights = weights.squeeze()

            if torch.cuda.is_available():
                res = torch.zeros(maps.size()[-2:]).cuda()
            else:
                res = Variable(torch.zeros(maps.size()[-2:]))

            for i, w in enumerate(weights):
                res += w * maps[i]
            return res

        # We need to re-assemble the architecture
        branch = torch.nn.Sequential(net.branch.block1,
                                     torch.nn.MaxPool2d(2),
                                     net.branch.block2,
                                     torch.nn.MaxPool2d(2),
                                     net.branch.block3)
        o_l = branch(l).data
        o_m = branch(m).data
        # After extracting the features, we weigh them based on the provided weights
        o_l = weigh_maps(wl, o_l)
        o_m = weigh_maps(wm, o_m)
        return F.relu(o_l), F.relu(o_m)


# Loading models upon importing
ENSEMBLE = _KneeNetEnsemble()


def get_result(img_path):
    """ Given path to an img, return prediction result and heatmap
        :param img_path: A str to img
        :returns:
            probs_df: A Dataframe with probability result
            cam: A Image with attention map drawn. Pillow image
            orig: Original image. Pillow image.
    """
    orig, heatmap, probs = ENSEMBLE.predict(img_path)
    probs_df = DataFrame(probs, index=['probs'],
                         columns=['KL{}'.format(i) for i in range(5)])

    # Draw heatmap
    heatmap = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.
    orig_np = np.array(orig) / 255.
    orig_np = np.repeat(np.expand_dims(orig_np, -1), 3, -1)
    cam = heatmap + orig_np
    cam = np.uint8(255 * cam / np.max(cam))

    # Swap R and B for better visualization
    cam = Image.fromarray(np.stack([cam[:, :, 2], cam[:, :,1], cam[:, :, 0]], -1))
    return probs_df, cam, orig
