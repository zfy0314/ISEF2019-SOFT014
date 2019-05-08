# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
from numpy import linalg as la
import os
import yaml
import gensim
import json
import pickle
#from six.moves import cPickle as pickle

import torch

from core.config import cfg
from core.test_rel import im_detect_rels
from datasets import task_evaluation_rel as task_evaluation
from datasets.json_dataset_rel import JsonDataset
from modeling import model_builder_rel
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
from utils.io import save_object
from utils.timer import Timer

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    # Generic case that handles all network types other than RPN-only nets
    # and RetinaNet
    child_func = test_net
    parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]
    proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            logger.info('pid {}: using parent case'.format(os.getpid()))
            all_results = []
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.append(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            logger.info('pid {}: using child case'.format(os.getpid()))
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb(gt=args.do_val))
        all_results = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_results = test_net(
            args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    
    logger.info('Starting evaluation now...')
    task_evaluation.eval_rel_results(all_results, output_dir, args.do_val)
    
    return all_results


def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]
        
    if args.do_val:
        opts += ['--do_val']
    if args.use_gt_boxes:
        opts += ['--use_gt_boxes']
        
    if args.use_gt_labels:
        opts += ['--use_gt_labels']

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'rel_detection', num_images, binary, output_dir,
        args.load_ckpt, args.load_detectron, opts
    )

    # Collate the results from each subprocess
    all_results = []
    for det_data in outputs:
        all_results += det_data
    
    if args.use_gt_boxes:
        if args.use_gt_labels:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_prdcls.pkl')
        else:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_sgcls.pkl')
    else:
        det_file = os.path.join(args.output_dir, 'rel_detections.pkl')
    save_object(all_results, det_file)
    logger.info('Wrote rel_detections to: {}'.format(os.path.abspath(det_file)))

    return all_results


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range, args.do_val
    )
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    
    num_images = len(roidb)
    all_results = [None for _ in range(num_images)]
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        box_proposals = None
            
        im = cv2.imread(entry['image'])
        if args.use_gt_boxes:
            im_results = im_detect_rels(model, im, dataset_name, box_proposals, timers, entry, args.use_gt_labels)
        else:
            im_results = im_detect_rels(model, im, dataset_name, box_proposals, timers)
        
        im_results.update(dict(image=entry['image']))
        # add gt
        if args.do_val:
            im_results.update(
                dict(gt_sbj_boxes=entry['sbj_gt_boxes'],
                     gt_sbj_labels=entry['sbj_gt_classes'],
                     gt_obj_boxes=entry['obj_gt_boxes'],
                     gt_obj_labels=entry['obj_gt_classes'],
                     gt_prd_labels=entry['prd_gt_classes']))
        
        all_results[i] = im_results

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (timers['im_detect_rels'].average_time)
            logger.info((
                'im_detect: range [{:d}, {:d}] of {:d}: '
                '{:d}/{:d} {:.3f}s (eta: {})').format(
                start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                start_ind + num_images, det_time, eta))

    logger.info('finish inferencing all images.')
    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'rel_detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        if args.use_gt_boxes:
            if args.use_gt_labels:
                det_name = 'rel_detections_gt_boxes_prdcls.pkl'
            else:
                det_name = 'rel_detections_gt_boxes_sgcls.pkl'
        else:
            det_name = 'rel_detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(all_results, det_file)
    logger.info('Wrote rel_detections to: {}'.format(os.path.abspath(det_file)))
    return all_results


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder_rel.Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range, do_val=True):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb(gt=do_val)

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def zfy_inference(
        args,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    #imgs = os.listdir('/home/zfy/Data/projects/isef/Large-Scale-VRD/data/vrd/train_images')
    imgs = ["test.jpg"]
    finals = []
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    dataset_name, proposal = get_inference_dataset(0)
    for xyz in imgs:
        im_file = os.path.join('/home/zfy/Data/projects/isef/Large-Scale-VRD/', xyz)
        #im_file = xyz
        timers = defaultdict(Timer)
        box_proposals = None      
        im = cv2.imread(im_file)
        
        
        im_results = im_detect_rels(model, im, dataset_name, box_proposals, timers)
        im_results.update(dict(image=im_file))

        det_boxes_sbj = im_results['sbj_boxes']  # (#num_rel, 4)
        det_boxes_obj = im_results['obj_boxes']  # (#num_rel, 4)
        det_labels_sbj = im_results['sbj_labels']  # (#num_rel,)
        det_labels_obj = im_results['obj_labels']  # (#num_rel,)
        det_scores_sbj = im_results['sbj_scores']  # (#num_rel,)
        det_scores_obj = im_results['obj_scores']  # (#num_rel,)
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
        print('finished inferencing for {}'.format(im_file))
        finals.append(out_dict)

    with open('sg-out.pkl', 'wb') as fout:
        pickle.dump(finals, fout, pickle.HIGHEST_PROTOCOL)
    return finals

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))
