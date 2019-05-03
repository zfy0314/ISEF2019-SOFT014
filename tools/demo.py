import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pprint
import sys
import time
from six.moves import cPickle as pickle

import torch
from torchvision import transforms 

import _init_paths 
from caption.model import Vocabulary
from cpation.model import EncoderCNN, DecoderRNN
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine_rel import run_inference
from merge import get_merged
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='caption demo')

    parser.add_argument(
        '--image', type=str, required=True, 
        help='input image for generating caption')
    parser.add_argument(
        '--encoder_path', type=str, default='models/encoder-16-4000.ckpt', 
        help='path for trained encoder')
    parser.add_argument(
        '--decoder_path', type=str, default='models/decoder-16-4000.ckpt', 
        help='path for trained decoder')
    parser.add_argument(
        '--vocab_path', type=str, default='data/vocab.pkl', 
        help='path for vocabulary wrapper')
    parser.add_argument(
        '--embed_size', type=int , default=256, 
        help='dimension of word embedding vectors')
    parser.add_argument(
        '--hidden_size', type=int , default=512, 
        help='dimension of lstm hidden states')
    parser.add_argument(
        '--num_layers', type=int , default=1, 
        help='number of layers in lstm')

    parser.add_argument(
        '--dataset', default='vg',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')
    parser.add_argument(
        '--output_dir', default='outputs/'
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')
    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')
    parser.add_argument(
        '--do_val', dest='do_val', help='do evaluation', action='store_true')
    parser.add_argument(
        '--use_gt_boxes', dest='use_gt_boxes', help='use gt boxes for sgcls/prdcls', action='store_true')
    parser.add_argument(
        '--use_gt_labels', dest='use_gt_labels', help='use gt boxes for sgcls/prdcls', action='store_true')

    return parser.parse_args()

def get_base(args):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN(args.embed_size).eval()
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    image = Image.open(args.image)
    image = image.resize([224, 224], Image.LANCZOS)
    image = transform(image).unsqueeze(0)
    image_tensor = image.to(device)

    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    print (sentence)
    return sentence

def get_sg(args):
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    cfg.TEST.DATASETS = ('vg_val',)
    cfg.MODEL.NUM_CLASSES = 151
    cfg.MODEL.NUM_PRD_CLASSES = 50  # exclude background

    if not cfg.MODEL.RUN_BASELINE:
        assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
            'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    args.test_net_file, _ = os.path.splitext(__file__)
    args.cuda = True

    timers = defaultdict(Timer)
    box_proposals = None      
    im = cv2.imread(args.image)
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    dataset_name, proposal = get_inference_dataset(0)
    im_results = im_detect_rels(model, im, dataset_name, box_proposals, timers)
    im_results.update(dict(image=im_file))

    det_boxes_sbj = im_results['sbj_boxes']
    det_boxes_obj = im_results['obj_boxes']
    det_labels_sbj = im_results['sbj_labels']
    det_labels_obj = im_results['obj_labels']
    det_scores_sbj = im_results['sbj_scores']
    det_scores_obj = im_results['obj_scores']
    det_scores_prd = im_results['prd_scores'][:, 1:]

    det_labels_prd = np.argsort(-det_scores_prd, axis=1)
    det_scores_prd = -np.sort(-det_scores_prd, axis=1)

    det_scores_so = det_scores_sbj * det_scores_obj
    det_scores_spo = det_scores_so[:, None] * det_scores_prd[:, :2]
    det_scores_inds = argsort_desc(det_scores_spo)[:100]
    det_scores_top = det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]
    det_boxes_so_top = np.hstack(
        (det_boxes_sbj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))
    det_labels_p_top = det_labels_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]]
    det_labels_spo_top = np.vstack(
        (det_labels_sbj[det_scores_inds[:, 0]], det_labels_p_top, det_labels_obj[det_scores_inds[:, 0]])).transpose() 
                    
    det_boxes_s_top = det_boxes_so_top[:, :4]
    det_boxes_o_top = det_boxes_so_top[:, 4:]
    det_labels_s_top = det_labels_spo_top[:, 0]
    det_labels_p_top = det_labels_spo_top[:, 1]
    det_labels_o_top = det_labels_spo_top[:, 2]
    out_dict = {}
    out_dict['boxes_s_top'] = det_boxes_s_top
    out_dict['boxes_o_top'] = det_boxes_o_top
    out_dict['labels_s_top'] = det_labels_s_top
    out_dict['labels_p_top'] = det_labels_p_top
    out_dict['labels_o_top'] = det_labels_o_top
    out_dict['scores_top'] = det_scores_top
    out_dict['image'] = im_file

    with open(os.path.join(args.output_dir, 'test-out.pkl'), 'wb') as fout:
        pickle.dump(out_dict, fout, pickle.HIGHEST_PROTOCOL)
    return out_dict

def main():
    
    args = parse_args()

    raw = {'sentence': '', 'graph' = {}}
    p = Pool(2)
    raw['sentence'] = p.apply_async(get_base, (, ))
    raw['graph'] = p.apply_async(get_sg, (, ))
    p.close()
    p.join()

    result = {}
    for key in raw.keys():
        result[key] = raw[key].get()

    enhanced = get_merged()

    print('scene-graph enhanced caption: {}'.format(enhanced))
