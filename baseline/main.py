# -*- coding: utf-8 -*-
#########################################################################
# This file is derived from Curious AI/mean-teacher, under the Creative Commons Attribution-NonCommercial
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

import argparse
import json
import os
import time

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from utils import ramps
from DatasetDcase2019Task4 import DatasetDcase2019Task4
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from utils.Scaler import Scaler
from TestModel import test_model
from evaluation_measures import get_f_measure_by_class, get_predictions, audio_tagging_results, compute_strong_metrics
from models.CRNN import CRNN
import config as cfg
from utils.utils import ManyHotEncoder, create_folder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms, AverageMeterSet, EarlyStopping
from utils.Logger import LOG

np.random.seed(5)
torch.manual_seed(0)

def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1.):
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    beta1 = rampdown_value * cfg.beta1_after_rampdown + (1. - rampdown_value) * cfg.beta1_before_rampdown
    beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, beta2)
        param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, optimizer, epoch, ema_model=None, weak_mask=None, strong_mask=None):
    """ One epoch of a Mean Teacher model
    :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
    Should return 3 values: teacher input, student input, labels
    :param model: torch.Module, model to be trained, should return a weak and strong prediction
    :param optimizer: torch.Module, optimizer used to train the model
    :param epoch: int, the current epoch of training
    :param ema_model: torch.Module, student model, should return a weak and strong prediction
    :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
    :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
    """
    class_criterion = nn.BCELoss()
    consistency_criterion_strong = nn.MSELoss()
    [class_criterion, consistency_criterion_strong] = to_cuda_if_available(
        [class_criterion, consistency_criterion_strong])

    meters = AverageMeterSet()

    LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    rampup_length = len(train_loader) * 50 # cfg.n_epoch // 2
    for i, (batch_input, ema_batch_input, target) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + i
        if global_step < rampup_length:
            rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        else:
            rampup_value = 1.0

        # Todo check if this improves the performance
        # adjust_learning_rate(optimizer, rampup_value) #, rampdown_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        [batch_input, ema_batch_input, target] = to_cuda_if_available([batch_input, ema_batch_input, target])
        LOG.debug(batch_input.mean())
        # Outputs
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()

        strong_pred, weak_pred = model(batch_input)
        loss = None
        # Weak BCE Loss
        # Take the max in the time axis
        target_weak = target.max(-2)[0]
        if weak_mask is not None:
            weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
            ema_class_loss = class_criterion(weak_pred_ema[weak_mask], target_weak[weak_mask])

            if i == 0:
                LOG.debug("target: {}".format(target.mean(-2)))
                LOG.debug("Target_weak: {}".format(target_weak))
                LOG.debug("Target_weak mask: {}".format(target_weak[weak_mask]))
                LOG.debug(weak_class_loss)
                LOG.debug("rampup_value: {}".format(rampup_value))
            meters.update('weak_class_loss', weak_class_loss.item())

            meters.update('Weak EMA loss', ema_class_loss.item())

            loss = weak_class_loss

        # Strong BCE loss
        if strong_mask is not None:
            strong_class_loss = class_criterion(strong_pred[strong_mask], target[strong_mask])
            meters.update('Strong loss', strong_class_loss.item())

            strong_ema_class_loss = class_criterion(strong_pred_ema[strong_mask], target[strong_mask])
            meters.update('Strong EMA loss', strong_ema_class_loss.item())
            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:

            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update('Consistency weight', consistency_cost)
            # Take only the consistence with weak and unlabel
            consistency_loss_strong = consistency_cost * consistency_criterion_strong(strong_pred,
                                                                                      strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong

            meters.update('Consistency weight', consistency_cost)
            # Take only the consistence with weak and unlabel
            consistency_loss_weak = consistency_cost * consistency_criterion_strong(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start

    LOG.info(
        'Epoch: {}\t'
        'Time {:.2f}\t'
        '{meters}'.format(
            epoch, epoch_time, meters=meters))


if __name__ == '__main__':

    LOG.info("MEAN TEACHER")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-ns", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                        help="Not using synthetic labels during training")

    parser.add_argument("-nw", '--no_weak', dest='no_weak', action='store_true', default=False,
                        help="Not using weak label during training")

    parser.add_argument("-m", '--message', dest='message', type=str, default="No Message",
                        help="Message printed on top of logs")

    parser.add_argument("-ts", '--time_shifting', dest='time_shift', action='store_true', default=False,
                        help="using time shifting for data augmentation")

    parser.add_argument("-ft", '--frequency_trunc', dest='frequency_trunc', action='store_true', default=False,
                        help="using time shifting for data augmentation")

    parser.add_argument("-gn", '--gaussian_noise', dest='gaussian_noise', action='store_true', default=False,
                        help="using time shifting for data augmentation")

    parser.add_argument("-wa", '--weak_augmentation', dest='weak_augmentation', action='store_true', default=False,
                        help="using time shifting for data augmentation")

    parser.add_argument("-ua", '--unlabel_augmentation', dest='unlabel_augmentation', action='store_true', default=False,
                        help="using time shifting for data augmentation")

    parser.add_argument("-sa", '--strong_augmentation', dest='strong_augmentation', action='store_true', default=False,
                        help="using time shifting for data augmentation")

    parser.add_argument("-d", '--data_multiplier', type=int, default=1, dest="data_multiplier",
                         help="int Multiplier of dataLoad len.")

    f_args = parser.parse_args()
    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic
    no_weak = f_args.no_weak
    data_multiplier = f_args.data_multiplier

    weak_augmentation = f_args.weak_augmentation
    unlabel_augmentation = f_args.unlabel_augmentation
    strong_augmentation = f_args.strong_augmentation

    time_shifting = f_args.time_shift
    frequency_trunc=f_args.frequency_trunc
    gaussian_noise=f_args.gaussian_noise

    augmentations = []
    suffix_name_aug = ""
    if time_shifting:
        augmentations.append('time_shifting')
        suffix_name_aug += "_ts"
    if frequency_trunc:
        augmentations.append('frequency_trunc')
        suffix_name_aug += "_ft"
    if gaussian_noise:
        augmentations.append('gaussian_noise')
        suffix_name_aug += "_gn"

    message = f_args.message

    LOG.info("subpart_data = {}".format(reduced_number_of_data))
    LOG.info("Using synthetic data = {}".format(not no_synthetic))
    LOG.info("-- DATA AUGMENTATION --")
    LOG.info("Data_multiplier = {}".format(data_multiplier))
    LOG.info("Time shifting = {}".format(time_shifting))
    LOG.info("Frequency trunc = {}".format(frequency_trunc))
    LOG.info("Gaussian noise = {}".format(gaussian_noise))
    LOG.info("--With--")
    LOG.info("weak data augmentaton = {}".format(weak_augmentation))
    LOG.info("unlabel data augmentaton = {}".format(unlabel_augmentation))
    LOG.info("strong data augmentaton = {}".format(strong_augmentation))
    LOG.info("MESSAGE: " + str(message))

    if no_synthetic and no_weak:
        add_dir_model_name = "_no_synthetic_no_weak"
    elif no_synthetic:
        add_dir_model_name = "_no_synthetic"
    elif no_weak:
        add_dir_model_name = "_no_weak"
    else:
        add_dir_model_name = "_with_synth&weak"

    store_dir = os.path.join("stored_data", "MeanTeacher" + add_dir_model_name + suffix_name_aug + "_" +
                             str(data_multiplier))
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    create_folder(store_dir)
    create_folder(saved_model_dir)
    create_folder(saved_pred_dir)

    pooling_time_ratio = cfg.pooling_time_ratio  # --> Be careful, it depends of the model time axis pooling
    # ##############
    # DATA
    # ##############
    dataset = DatasetDcase2019Task4(cfg.workspace,
                                    base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                                    save_log_feature=False)

    weak_df = dataset.initialize_and_get_df(cfg.weak, reduced_number_of_data)
    unlabel_df = dataset.initialize_and_get_df(cfg.unlabel, reduced_number_of_data)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = dataset.initialize_and_get_df(cfg.synthetic, reduced_number_of_data, download=False)
    validation_df = dataset.initialize_and_get_df(cfg.validation, reduced_number_of_data)

    classes = cfg.classes
    many_hot_encoder = ManyHotEncoder(classes, n_frames=cfg.max_frames // pooling_time_ratio)

    transforms = get_transforms(cfg.max_frames)

    if no_synthetic and no_weak:
        frac_synth = 0
        frac_weak = 0
    elif no_synthetic and not no_weak:
        frac_synth = 0
        frac_weak = 1
    elif no_weak and not no_synthetic:
        frac_weak = 0
        frac_synth = 1
    else:
        frac_weak = 0.5
        frac_synth = 0.5

    # Divide weak in train and valid

    train_weak_df = weak_df.sample(frac=frac_weak, random_state=26)
    # valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    # LOG.debug(valid_weak_df.event_labels.value_counts())

    # Divide synthetic in train and valid
    valid_synth_df = synthetic_df.loc[synthetic_df.filename.drop_duplicates().sample(n=470, random_state=26).index]
    filenames_train = synthetic_df.drop(valid_synth_df.index).sample(n=1578).reset_index(drop=True)
    filenames_train = filenames_train.sample(frac=frac_synth).reset_index(drop=True)
    # filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=frac_synth, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    # valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)
    LOG.info("size valid synth df : {}".format(valid_synth_df.shape))
    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    # LOG.debug(valid_synth_df.event_label.value_counts())

    # Normalize
    train_weak_data_norm = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                 transform=transforms)
    unlabel_data_norm = DataLoadDf(unlabel_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                   transform=transforms)
    train_synth_data_norm = DataLoadDf(train_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms)
    list_dataset_norm = [unlabel_data_norm]
    if not no_synthetic:
        list_dataset_norm.append(train_synth_data_norm)
    if not no_weak:
        list_dataset_norm.append(train_weak_data_norm)


    scaler = Scaler()
    scaler.calculate_scaler(ConcatDataset(list_dataset_norm))

    LOG.debug(scaler.mean_)

    if weak_augmentation:
        train_weak_data = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df, augmentations, data_multiplier,
                                 transform=transforms)
    else:
        train_weak_data = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df, transform=transforms)
    if unlabel_augmentation:
        unlabel_data = DataLoadDf(unlabel_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df, augmentations, data_multiplier,
                              transform=transforms)
    else:
        unlabel_data = DataLoadDf(unlabel_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df, transform=transforms)
    if strong_augmentation:
        train_synth_data = DataLoadDf(train_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df, augmentations, data_multiplier,
                                  transform=transforms)
    else:
        train_synth_data = DataLoadDf(train_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                      transform=transforms)


    if not no_synthetic:
        if not no_weak:
            list_dataset = [train_weak_data, unlabel_data, train_synth_data]
            batch_sizes = [cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4]
            strong_mask = slice(cfg.batch_size//4 + cfg.batch_size//2, cfg.batch_size)
            weak_mask = slice(batch_sizes[0])
        else:
            list_dataset = [unlabel_data, train_synth_data]
            batch_sizes = [cfg.batch_size // 2, cfg.batch_size // 2]
            strong_mask = slice(cfg.batch_size // 2, cfg.batch_size)
            weak_mask = None
    else:
        if not no_weak:
            list_dataset = [train_weak_data, unlabel_data]
            batch_sizes = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
            strong_mask = None
            weak_mask = slice(batch_sizes[0])
        else:
            list_dataset = [unlabel_data]
            batch_sizes = [cfg.batch_size]
            strong_mask = None
            weak_mask = None



    # Assume weak data is always the first one
    # weak_mask = slice(batch_sizes[0])

    transforms = get_transforms(cfg.max_frames, scaler, augment_type="noise")
    for i in range(len(list_dataset)):
        list_dataset[i].set_transform(transforms)

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset,
                                      batch_sizes=batch_sizes)

    training_data = DataLoader(concat_dataset, batch_sampler=sampler)

    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)
    valid_synth_data = DataLoadDf(valid_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                   transform=transforms_valid)


    # Eval 2018
    eval_2018_df = dataset.initialize_and_get_df(cfg.eval2018, reduced_number_of_data)
    eval_2018 = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                           transform=transforms_valid)
    valid_weak_df = eval_2018_df.sample(n=500)
    valid_weak_df = valid_weak_df.reset_index(drop=True)

    valid_weak_data = DataLoadDf(valid_weak_df, dataset.get_feature_file, many_hot_encoder.encode_weak,
                                 transform=transforms_valid)

    # ##############
    # Model
    # ##############
    no_load = True
    init_crnn = "stored_data/init_crnn"
    # if os.path.exists(init_crnn):
    #     try:
    #         state = torch.load(init_crnn, map_location="cpu")
    #         crnn_kwargs = state["model"]["kwargs"]
    #         crnn = CRNN(**crnn_kwargs)
    #         crnn.load(parameters=state["model"]["state_dict"])
    #         crnn_ema = CRNN(**crnn_kwargs)
    #         crnn_ema.load(parameters=state["model_ema"]["state_dict"])
    #
    #         optim_kwargs = {"lr": cfg.lr, "betas": (0.9, 0.999)}
    #         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    #         pooling_time_ratio = state["pooling_time_ratio"]
    #         no_load = False
    #     except (RuntimeError, TypeError) as e:
    #         LOG.warn("Init model couldn't be load, rewritting the file")
    if no_load:
        crnn_kwargs = cfg.crnn_kwargs
        crnn = CRNN(**crnn_kwargs)
        crnn_ema = CRNN(**crnn_kwargs)

        optim_kwargs = {"lr": cfg.max_learning_rate, "betas": (cfg.beta1_after_rampdown, cfg.beta2_after_rampup)}
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
        bce_loss = nn.BCELoss()

        state = {
            'model': {"name": crnn.__class__.__name__,
                      'args': '',
                      "kwargs": crnn_kwargs,
                      'state_dict': crnn.state_dict()},
            'model_ema': {"name": crnn_ema.__class__.__name__,
                          'args': '',
                          "kwargs": crnn_kwargs,
                          'state_dict': crnn_ema.state_dict()},
            'optimizer': {"name": optimizer.__class__.__name__,
                          'args': '',
                          "kwargs": optim_kwargs,
                          'state_dict': optimizer.state_dict()},
            "pooling_time_ratio": pooling_time_ratio,
            "scaler": scaler.state_dict(),
            "many_hot_encoder": many_hot_encoder.state_dict()
        }

        crnn.apply(weights_init)
        crnn_ema.apply(weights_init)
        torch.save(state, init_crnn)

    LOG.info(crnn)

    for param in crnn_ema.parameters():
        param.detach_()

    save_best_cb = SaveBest("sup")
    early_stopping = EarlyStopping(crnn, 50, val_comp="sup")

    # ##############
    # Train
    # ##############
    for epoch in range(cfg.n_epoch):
        crnn = crnn.train()
        crnn_ema = crnn_ema.train()

        [crnn, crnn_ema] = to_cuda_if_available([crnn, crnn_ema])

        train(training_data, crnn, optimizer, epoch, ema_model=crnn_ema, weak_mask=weak_mask, strong_mask=strong_mask)

        crnn = crnn.eval()
        LOG.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn, valid_synth_data, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      save_predictions=None)
        valid_events_metric = compute_strong_metrics(predictions, valid_synth_df)

        LOG.info("\n ### Valid weak metric ### \n")
        weak_metric = get_f_measure_by_class(crnn, len(classes),
                                             DataLoader(valid_weak_data, batch_size=cfg.batch_size))

        LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
        LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

        state['model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_events_metric.results()
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            if not no_synthetic:
                global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
                global_valid = global_valid + np.mean(weak_metric)
            else:
                global_valid = np.mean(weak_metric)
            if save_best_cb.apply(global_valid):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)
        if early_stopping.apply(valid_events_metric.results()["class_wise_average"]["f_measure"]["f_measure"]):
            LOG.info("\n\n\nEARLY STOPPING\n\n\n")
            #break

    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        LOG.info("testing model: {}".format(model_fname))
    else:
        LOG.info("testing model of last epoch: {}".format(cfg.n_epoch))

    # ##############
    # Validation
    # ##############
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.csv")
    valid_metric = test_model(state, cfg.validation,reduced_number_of_data, predicitons_fname)
    with open(os.path.join(store_dir, "results_validation.json"), "w") as f:
        json.dump(valid_metric.results(), f)

    # with open(os.path.join(store_dir, "results_validation.json"), "r") as f:
    #     valid_metric_res = json.load(f)
    # valid_metric_res["class_wise"]
    # valid_metric_res["class_wise_average"]
