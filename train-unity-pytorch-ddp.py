#https://github.com/inkawhich/pt-distributed-tutorial/blob/master/pytorch-aws-distributed-tutorial.py

import os
import sys
import time
import logging
import shutil

from pathlib import Path

import torch
import torch.nn
import torch.nn.parallel
import torch.distributed

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Scantensus import AverageMeter
from Scantensus.utils.json import get_keypoint_names_and_colors_from_json
from ScantensusPT.datasets.unity import UnityDataset, UnityMakeHeatmaps
#from ScantensusPT.optim.ranger import Ranger
from ScantensusPT.optim.radam import RAdam
from ScantensusPT.losses import MattMSEClampLoss, MattMSESumLoss, LVIDMetric

from Scantensus.nets.HRNet_CFG_K_Sigmoid import get_net_cfg
from ScantensusPT.nets.HRNetV2M7 import get_seg_model

#########

HOST = 'thready3'
PROJECT = 'unity'
EXPERIMENT = 'unity-147'

DEBUG = False
DDP_PORT = 23247

if DEBUG:
    SINGLE_TRAIN_WORKERS = 0
    SINGLE_VAL_WORKERS = 0
else:
    SINGLE_TRAIN_WORKERS = 5
    SINGLE_VAL_WORKERS = 5

##########

DOT_SD = 4
CURVE_SD = 2

DOT_WEIGHT_SD = DOT_SD * 5
CURVE_WEIGHT_SD = CURVE_SD * 5

DOT_WEIGHT = 80
CURVE_WEIGHT = 20

IMAGE_CROP_SIZE = (640, 640)
IMAGE_OUT_SIZE = (608, 608)

PRE_POST = False

####

INITIAL_LEARNING_RATE = 0.001
LR_PATIENCE = 20
LEN_MEMORY = 110
EPOCHS = 300

SINGLE_TRAIN_BATCH_SIZE = 6
SINGLE_VAL_BATCH_SIZE = 16

################
NUM_CUDA_DEVICES = torch.cuda.device_count()
CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', None)
################

################
RANK = int(sys.argv[1])
LOCAL_RANK = int(sys.argv[2])
WORLD_SIZE = int(sys.argv[3])
################

###############
if HOST == "thready1":
    USE_CUDA = True
    DISTRIBUTED_BACKEND = "nccl"
    DATA_DIR = Path("/") / "home" / "matthew" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "home" / "matthew" / "matt-output"
elif HOST == "thready3":
    USE_CUDA = True
    DISTRIBUTED_BACKEND = "nccl"
    DATA_DIR = Path("/") / "home" / "matthew" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "home" / "matthew" / "matt-output"
elif HOST == "matt-laptop":
    USE_CUDA = False
    DISTRIBUTED_BACKEND = "gloo"
    DATA_DIR = Path("/") / "Volumes" / "Matt-Data" / "Projects-Clone" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "Volumes" / "Matt-Temp" / "matt-output"
elif HOST == "rcs":
    USE_CUDA = True
    DISTRIBUTED_BACKEND = "nccl"
    DATA_DIR = Path(os.environ['PT_DATA_DIR'])
    OUTPUT_DIR = Path(os.environ['PT_OUTPUT_DIR'])
    EXPERIMENT = str(os.environ['PT_EXPERIMENT'])
else:
    raise Exception
################

########
PNG_CACHE_DIR = DATA_DIR / "png-cache"

JSON_KEYS_PATH = DATA_DIR / "labels" / PROJECT / "keys.json"
DB_TRAIN_PATH = DATA_DIR / "labels" / PROJECT / "labels-train.json"
DB_VAL_PATH = DATA_DIR / "labels" / PROJECT / "labels-val.json"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / PROJECT / EXPERIMENT
CHECKPOINT_KEYS_PATH = OUTPUT_DIR / "checkpoints" / PROJECT / EXPERIMENT / "keys.json"
LOG_DIR = OUTPUT_DIR / "logs" / PROJECT / EXPERIMENT
TXT_LOG_DIR = OUTPUT_DIR / "txt_logs" / PROJECT / EXPERIMENT
#########

#########
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TXT_LOG_DIR, exist_ok=True)
shutil.copy(JSON_KEYS_PATH, CHECKPOINT_KEYS_PATH)
############

#########
DATA_DIR = str(DATA_DIR)
DB_TRAIN_PATH = str(DB_TRAIN_PATH)
DB_VAL_PATH = str(DB_VAL_PATH)
###########


def reduce(value, device):
    value = torch.tensor(value, device=device)
    torch.distributed.all_reduce(value)
    return value.item()


def train(dataloader, make_heatmaps, model, loss_fn, optimizer, metric_fns, metric_names, epoch, writer=None):
    device = "cuda" if model.is_cuda else "cpu"

    model.train()

    loss_meter = AverageMeter()

    metric_epoch_vals = []

    metric_meters = []
    for _ in range(len(metric_fns)):
        metric_meters.append(AverageMeter())

    time_data_meter = AverageMeter()
    time_compute_meter = AverageMeter()

    time_end = time.time()

    for step, batch in enumerate(dataloader):
        metric_vals = []

        time_start = time.time()
        time_data_meter.update(time_start - time_end)

        x = batch.image.to(device=device, dtype=torch.float32, non_blocking=True).div(255.0).add(-0.5)

        heatmaps, weights = make_heatmaps(label_data=batch.label_data,
                                          label_height_shift=batch.label_height_shift,
                                          label_width_shift=batch.label_width_shift,
                                          transform_matrix=batch.transform_matrix)

        y_true = [heatmap.to(device=device, dtype=torch.float32, non_blocking=True).div(255.0) for heatmap in heatmaps]
        y_weights = [weight.to(device=device, dtype=torch.float32, non_blocking=True) for weight in weights]

        y_pred = model(x)

        loss = loss_fn(y_pred=y_pred, y_true=y_true, y_weights=y_weights)
        loss_val = loss.item()

        for metric_fn in metric_fns:
            metric_val = metric_fn(y_pred=y_pred, y_true=y_true, y_weights=y_weights).item()
            metric_vals.append(metric_val)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_meter.update(val=loss_val, n=x.size(0))

        metric_str = f""
        for metric_name, metric_val, metric_meter in zip(metric_names, metric_vals, metric_meters):
            metric_meter.update(val=metric_val, n=x.size(0))
            metric_str = metric_str + f"{metric_name}={metric_val:8.6f} "

        time_end = time.time()
        time_compute_meter.update(time_end - time_start)

        desc_str = f"Epoch {epoch} Train {step} "
        loss_str = f"Loss {loss_meter.val:8.6f} "
        time_str = f"Time D={time_data_meter.val:5.3f} C={time_compute_meter.val:5.3f}"
        log_str = desc_str + loss_str + metric_str + time_str

        train_logger.info(log_str)

    loss_epoch_val = reduce(value=loss_meter.avg, device=device) / WORLD_SIZE
    if writer is not None:
        writer.add_scalar(f"Loss/train", loss_epoch_val, epoch)

    metric_epoch_str = f""
    for metric_name, metric_meter in zip(metric_names, metric_meters):
        metric_epoch_val = reduce(value=metric_meter.avg, device=device) / WORLD_SIZE
        metric_epoch_vals.append(metric_epoch_val)
        metric_epoch_str = metric_epoch_str + f"{metric_name}={metric_epoch_val:8.6f} "
        if writer is not None:
            writer.add_scalar(f"{metric_name}/train", metric_epoch_val, epoch)

    desc_epoch_str = f"Epoch {epoch} Train final "
    loss_epoch_str = f"Loss {loss_epoch_val:8.6f} "
    time_epoch_str = f"Time D {time_data_meter.sum:5.3f} C {time_compute_meter.sum:5.3f}"
    log_epoch_str = desc_epoch_str + loss_epoch_str + metric_epoch_str + time_epoch_str

    train_logger.info(log_epoch_str)

    return loss_epoch_val


def test(dataloader, make_heatmaps, model, loss_fn, metric_fns, metric_names, epoch, writer=None):
    device = "cuda" if model.is_cuda else "cpu"

    model.eval()

    loss_meter = AverageMeter()

    metric_epoch_vals = []

    metric_meters = []
    for _ in range(len(metric_fns)):
        metric_meters.append(AverageMeter())

    time_data_meter = AverageMeter()
    time_compute_meter = AverageMeter()

    time_end = time.time()

    with torch.no_grad():

        for step, batch in enumerate(dataloader):
            metric_vals = []

            time_start = time.time()
            time_data_meter.update(time_start - time_end)

            x = batch.image.to(device=device, dtype=torch.float32, non_blocking=True).div(255.0).add(-0.5)

            heatmaps, weights = make_heatmaps(label_data=batch.label_data,
                                              label_height_shift=batch.label_height_shift,
                                              label_width_shift=batch.label_width_shift,
                                              transform_matrix=batch.transform_matrix)

            y_true = [heatmap.to(device=device, dtype=torch.float32, non_blocking=True).div(255.0) for heatmap in heatmaps]
            y_weights = [weight.to(device=device, dtype=torch.float32, non_blocking=True) for weight in weights]

            y_pred = model(x)

            loss = loss_fn(y_pred=y_pred, y_true=y_true, y_weights=y_weights)
            loss_val = loss.item()

            for metric_fn in metric_fns:
                metric_val = metric_fn(y_pred=y_pred, y_true=y_true, y_weights=y_weights).item()
                metric_vals.append(metric_val)

            loss_meter.update(val=loss_val, n=x.size(0))

            metric_str = f""
            for metric_name, metric_val, metric_meter in zip(metric_names, metric_vals, metric_meters):
                metric_meter.update(val=metric_val, n=x.size(0))
                metric_str = metric_str + f"{metric_name}={metric_val:8.6f} "

            time_end = time.time()
            time_compute_meter.update(time_end - time_start)

            desc_str = f"Epoch {epoch}, Test step {step} "
            loss_str = f"Loss {loss_meter.val:8.6f} "
            time_str = f"Time D={time_data_meter.val:5.3f} C={time_compute_meter.val:5.3f}"
            log_str = desc_str + loss_str + metric_str + time_str

            train_logger.info(log_str)

        loss_epoch_val = reduce(value=loss_meter.avg, device=device) / WORLD_SIZE
        if writer is not None:
            writer.add_scalar(f"Loss/val", loss_epoch_val, epoch)

        metric_epoch_str = f""
        for metric_name, metric_meter in zip(metric_names, metric_meters):
            metric_epoch_val = reduce(value=metric_meter.avg, device=device) / WORLD_SIZE
            metric_epoch_vals.append(metric_epoch_val)
            metric_epoch_str = metric_epoch_str + f"{metric_name}={metric_epoch_val:8.6f} "
            if writer is not None:
                writer.add_scalar(f"{metric_name}/val", metric_epoch_val, epoch)

        desc_epoch_str = f"Epoch {epoch} Test final "
        loss_epoch_str = f"Loss {loss_epoch_val:8.6f} "
        time_epoch_str = f"Time D {time_data_meter.sum:5.3f} C {time_compute_meter.sum:5.3f}"
        log_epoch_str = desc_epoch_str + loss_epoch_str + metric_epoch_str + time_epoch_str

        train_logger.info(log_epoch_str)

    return loss_epoch_val

####

if __name__ == '__main__':

    # Establish Local Rank and set device on this node

    if USE_CUDA:
        ddp_device_ids = [LOCAL_RANK]
        ddp_output_device = LOCAL_RANK
        torch.cuda.set_device(LOCAL_RANK)
        DEVICE = torch.device('cuda', LOCAL_RANK)
    else:
        ddp_device_ids = [None]
        ddp_output_device = None
        DEVICE = torch.device('cpu', 0)

    print("Setting up logging")

    ## Set up logging

    dataset_logger = logging.getLogger('dataset')
    dataset_logger.setLevel(logging.WARNING)

    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.INFO)

    h1 = logging.StreamHandler(sys.stdout)
    h2 = logging.FileHandler(TXT_LOG_DIR / f"log-{LOCAL_RANK}.txt")
    dataset_logger.addHandler(h1)
    dataset_logger.addHandler(h2)
    train_logger.addHandler(h1)
    train_logger.addHandler(h2)

    ### Get keys

    keypoint_names, keypoint_cols = get_keypoint_names_and_colors_from_json(JSON_KEYS_PATH)
    num_keypoints = len(keypoint_names)

    ### Set up tensorboard

    if LOCAL_RANK == 0:
        writer = SummaryWriter(str(LOG_DIR))
    else:
        writer = None

    train_logger.info("Logging setup finished")
    train_logger.info(f"Number of CUDA Devices: {NUM_CUDA_DEVICES}")

    ### Initialize Process Group

    train_logger.info("Starting torch distributed")
    torch.distributed.init_process_group(backend=DISTRIBUTED_BACKEND, init_method=f"tcp://localhost:{DDP_PORT}", rank=RANK, world_size=WORLD_SIZE)

    train_logger.info("Finished torch distributed")

    ### Get model

    net_cfg = get_net_cfg()

    net_cfg['DATASET'] = {}
    net_cfg['MODEL']['PRETRAINED'] = False
    net_cfg['DATASET']['NUM_CLASSES'] = len(keypoint_names)
    if PRE_POST:
        net_cfg['DATASET']['NUM_INPUT_CHANNELS'] = 3 * 3
    else:
        net_cfg['DATASET']['NUM_INPUT_CHANNELS'] = 1 * 3

    train_logger.info("Starting: Single Model")
    single_model = get_seg_model(cfg=net_cfg)

    if USE_CUDA:
        single_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model)

    torch.manual_seed(42)
    single_model.init_weights()
    train_logger.info("Finished: Single Model")

    ###

    train_logger.info("Starting: Distributed")
    single_model = single_model.to(DEVICE)
    model = torch.nn.parallel.DistributedDataParallel(single_model, device_ids=ddp_device_ids, output_device=ddp_output_device)
    #model = torch.nn.parallel.DistributedDataParallel(single_model, device_ids=None, output_device=None)

    train_logger.info("Finished: Distributing")

    ###

    train_logger.info("Starting: Metrics generation")

    keep_layers_plax = torch.zeros(len(keypoint_names), device=DEVICE)
    keep_layers_plax[keypoint_names.index('lv-pw-top')] = 1.0
    keep_layers_plax[keypoint_names.index('lv-pw-bottom')] = 1.0
    keep_layers_plax[keypoint_names.index('lv-ivs-top')] = 1.0
    keep_layers_plax[keypoint_names.index('lv-ivs-bottom')] = 1.0

    loss_fn = MattMSEClampLoss()
    #loss_fn = MattMSESumLoss()

    metric_fns = [
        MattMSEClampLoss().to(DEVICE),
        #MattMSESumLoss(keep_layers=keep_layers_plax).to(DEVICE),
        MattMSESumLoss().to(DEVICE),
        LVIDMetric(keypoint_names=keypoint_names, dot_sd=DOT_SD),
    ]

    metric_names = [
        'MattMSEClampLoss',
        #'MattSumLoss_PLAX',
        'MattSumLoss',
        'LVID',
    ]

    params = list(model.parameters())

    #optimizer = SGD(params=params, lr=INITIAL_LEARNING_RATE)
    #optimizer = Adam(params=params, lr=INITIAL_LEARNING_RATE)
    optimizer = RAdam(params=params, lr=INITIAL_LEARNING_RATE)
    #optimizer = DeepMemory(params=params, lr=INITIAL_LEARNING_RATE, len_memory=LEN_MEMORY)
    #optimizer = Ranger(params=params, lr=INITIAL_LEARNING_RATE, use_gc=False)

    scheduler = ReduceLROnPlateau(optimizer, patience=LR_PATIENCE, factor=0.2, verbose=True)
    #scheduler = OneCycleLR(optimizer, max_lr=INITIAL_LEARNING_RATE, div_factor=10, total_steps=EPOCHS)

    train_logger.info("Finished: Metrics generation")

    ###

    train_logger.info("Starting: DataLoader")

    train_data = UnityDataset(labels_path=DB_TRAIN_PATH,
                              png_cache_dir=PNG_CACHE_DIR,
                              keypoint_names=keypoint_names,
                              transform=True,
                              image_crop_size=IMAGE_CROP_SIZE,
                              image_out_size=IMAGE_OUT_SIZE,
                              pre_post=PRE_POST,
                              device="cpu",
                              name='train')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=SINGLE_TRAIN_BATCH_SIZE,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=SINGLE_TRAIN_WORKERS,
                                                   pin_memory=False,
                                                   sampler=train_sampler)

    train_make_heatmaps = UnityMakeHeatmaps(keypoint_names=keypoint_names,
                                            image_crop_size=IMAGE_CROP_SIZE,
                                            image_out_size=IMAGE_OUT_SIZE,
                                            heatmap_scale_factors=(4, 2),
                                            dot_sd=DOT_SD,
                                            curve_sd=CURVE_SD,
                                            dot_weight_sd=DOT_WEIGHT_SD,
                                            curve_weight_sd=CURVE_WEIGHT_SD,
                                            dot_weight=DOT_WEIGHT,
                                            curve_weight=CURVE_WEIGHT,
                                            sub_pixel=True,
                                            device=DEVICE)

    val_data = UnityDataset(labels_path=DB_VAL_PATH,
                            png_cache_dir=PNG_CACHE_DIR,
                            keypoint_names=keypoint_names,
                            transform=False,
                            image_crop_size=IMAGE_CROP_SIZE,
                            image_out_size=IMAGE_CROP_SIZE,
                            pre_post=PRE_POST,
                            device="cpu",
                            name='val')

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=SINGLE_VAL_BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=SINGLE_VAL_WORKERS,
                                                 pin_memory=False,
                                                 sampler=val_sampler)

    val_make_heatmaps = UnityMakeHeatmaps(keypoint_names=keypoint_names,
                                          image_crop_size=IMAGE_CROP_SIZE,
                                          image_out_size=IMAGE_CROP_SIZE,
                                          heatmap_scale_factors=(4, 2),
                                          dot_sd=DOT_SD,
                                          curve_sd=CURVE_SD,
                                          dot_weight_sd=DOT_WEIGHT_SD,
                                          curve_weight_sd=CURVE_WEIGHT_SD,
                                          dot_weight=DOT_WEIGHT,
                                          curve_weight=CURVE_WEIGHT,
                                          sub_pixel=True,
                                          device=DEVICE)

    train_logger.info("Finished: DataLoader")

    ###

    for epoch in range(1, EPOCHS+1):

        train_sampler.set_epoch(epoch)

        if writer is not None:
            writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(dataloader=train_dataloader,
                           make_heatmaps=train_make_heatmaps,
                           model=model,
                           loss_fn=loss_fn,
                           metric_fns=metric_fns,
                           metric_names=metric_names,
                           optimizer=optimizer,
                           epoch=epoch,
                           writer=writer)

        torch.cuda.empty_cache()

        val_loss = test(dataloader=val_dataloader,
                        make_heatmaps=val_make_heatmaps,
                        model=model,
                        loss_fn=loss_fn,
                        metric_fns=metric_fns,
                        metric_names=metric_names,
                        epoch=epoch,
                        writer=writer)

        torch.cuda.empty_cache()

        if epoch % 10 == 0 and RANK == 0:
            train_logger.info(f"Starting: Checkpoint save")
            checkpoint_path = CHECKPOINT_DIR / f"weights-{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Finished: Checkpoing save")

        scheduler.step(metrics=val_loss)
        #scheduler.step()