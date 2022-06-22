# Group Author info:
# psomash Prakruthi Somashekarappa 
# rbraman Radhika B Raman
# srames22 Srivatsan Ramesh
#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import numpy as np
import time

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config

import torch.distributed as dist
import random

# Helper classes that will enable partitioning of the data into 50-50 and testing will be carried on with each node's respective data
""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234): # sizes is overridden with 0.5 0.5
        #print(sizes)
        self.data = data
        self.partitions = []
        #rng = random()
        random.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def evaluate_model_file(model_path, weights_path, img_path, class_names, batch_size=8, img_size=416,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True, rank=0):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    # To obtain the data loading time
    data_load_start_time = time.time()

    #print("Inside evaluate_model")

    dataloader = _create_validation_data_loader(
        img_path, batch_size, img_size, n_cpu)
    print("Total data loading time: %s seconds" % (time.time() - data_load_start_time))

    model = load_model(model_path, weights_path)
    
    # To obtain the inference time
    inference_start_time = time.time()  

    if(rank == 1):
        _evaluate(model,dataloader,class_names,img_size,iou_thres,conf_thres,nms_thres,verbose,rank)

    elif rank==0:
        metrics_output = _evaluate(model,dataloader,class_names,img_size,iou_thres,conf_thres,nms_thres,verbose,rank)
        print("Total inference time: %s seconds" % (time.time() - inference_start_time))
        return metrics_output


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose, rank):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    
    # if(rank==0):
    #     print("len of labels at 0")
    #     print(len(labels))
    # elif(rank==1):
    #     print("len of labels at 1")
    #     print(len(labels))

    #tensor_sample = torch.tensor(sample_metrics)
    #print(type(tensor_sample))    
        

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    
    #print("True Positives")
    #print(true_positives)
    
    # Idea: Since after each node completes testing on its respective half of data and true_positives, pred_scores, and pred_labels are used to compute
    # AP and in turn mAP, these values are passed to rank 0 since only rank 0 must perform computations. 
    
    # Message passing can only take place with tensors and hence converting the numPy arrays into a tensor on the device 
    pred_labels_tensor = torch.from_numpy(pred_labels).cuda(device)
    pred_scores_tensor = torch.from_numpy(pred_scores).cuda(device)
    true_pos_tensor = torch.from_numpy(true_positives).cuda(device)

    # labels is a list and hence does not need from_numpy
    labels_tensor = torch.tensor(labels).cuda(device)

    #print("PRED LABEL", max(pred_labels))

    # Rank 1 sends the necessary tensors to rank 0 i.e pred_labels, pred_scores, true_positives, labels all in tensor form
    if (rank == 1):
        send_x = torch.from_numpy(pred_labels).cuda(device)
        torch.distributed.send(send_x, dst=0)
        #print("pred label shape: ", pred_labels_tensor.shape)

        send_y = torch.from_numpy(pred_scores).cuda(device)
        torch.distributed.send(send_y, dst=0)
        #print("pred scores shape: ",pred_scores_tensor.shape)

        send_z = torch.from_numpy(true_positives).cuda(device)
        torch.distributed.send(send_z, dst=0)
        #print("true pos shape: ",true_pos_tensor.shape)

        send_l = torch.tensor(labels).cuda(device)
        torch.distributed.send(send_l, dst=0)

    # Rank 0 receives tensors from rank 1. To recieve them initial tensors buffers of their respective receiving tensors size (All tensors from rank 1 have the same shape)
    else:
        #n = np.zeros(pred_labels_tensor.shape)
        recv_x = torch.zeros(torch.Size([76732])).cuda(device)
        recv_y = torch.zeros(torch.Size([76732])).cuda(device)
        recv_z = torch.ones(torch.Size([76732]), dtype=torch.float64).cuda(device)
        recv_l = torch.zeros(17925).cuda(device)
        # print("local tensor ============== ")
        # print(recv_x)
        # print(recv_y)
        # print(recv_z)
        
        # Receiving the tensors from rank 1
        torch.distributed.recv(recv_x, src=1)

        torch.distributed.recv(recv_y, src=1)

        #torch.cuda.synchronize(torch.cuda.current_device())
        torch.distributed.recv(recv_z, src=1)

        torch.distributed.recv(recv_l, src=1)
    
    # Once the tensors have been received, the tensors are converted back to numpy which can be done only on the CPU
    if rank == 0:
                
        #print("recv tensor =============== ")
        recv_x = recv_x.to("cpu")
        recv_x = recv_x.numpy()

        recv_y = recv_y.to("cpu")
        recv_y = recv_y.numpy()

        recv_z = recv_z.to("cpu")
        recv_z = recv_z.numpy()


        recv_l = recv_l.to("cpu")
        recv_l = recv_l.tolist()

        # print(recv_x)
        # print(recv_y)
        # print(recv_z)

        # The converted tensors are now merged with rank 0's pred_scores, pred_labels, labels, true_positives
        pred_labels = np.concatenate((pred_labels, recv_x))

        # print("len of pred labels after concat")
        # print(len(pred_labels))
        pred_scores = np.concatenate((pred_scores, recv_y))
        true_positives = np.concatenate((true_positives, recv_z))
        # print("TP")
        # print(true_positives)

        # To append one list to another
        labels.extend(recv_l)

        # The numPy arrays have been appended and all image's array values are now in rank 0 and computation of AP and other metrics takes place
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        print_eval_stats(metrics_output, class_names, verbose)

        return metrics_output


def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    #print("Inside create validation")
    # print("len of dataset")
    # print(len(dataset))
    # To get the total number of participating nodes in the group
    size = dist.get_world_size() 
    bsz = 128 / size
    #print(bsz)
    # 0.5, 0.5 are the partition sizes and data is partitioned exactly into two and provided by the DataLoader to each node as batches according to the batch size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())

    # print("dist.get_rank()")
    # print(dist.get_rank())
    # print("len of partition")
    # print(len(partition))
    # train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)

    # Instead of passing the entire "dataset" the "partition" variable is being passed to the DataLoader
    dataloader = DataLoader(
        partition,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    #print(len(dataloader))
    return dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Load configuration from data file
    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # List of class names


    # distributed communication- Assuming MASTER_ADDR will be set up. 
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    ip_addr = os.environ['MASTER_ADDR']
    size = int(os.environ['SLURM_NTASKS'])
    #print("rank : %d" %rank)
    #print("size : %d" %size)
    backend = 'nccl'
    method = 'tcp://' + ip_addr + ":22233"

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #print("Before init")
    # Initializing the distribution process group and the distributed package
    dist.init_process_group(backend, world_size=size, rank=rank, init_method=method)

    #print("After init")
    
    # Since only root node must compute the values and only testing must be done on both nodes, there is no need for return values for rank 1 as it is going to 
    # send its tested values as thorugh message passing. 
    if rank==1:
        evaluate_model_file(args.model,args.weights,valid_path,class_names,batch_size=args.batch_size,img_size=args.img_size,n_cpu=args.n_cpu,iou_thres=args.iou_thres,conf_thres=args.conf_thres,nms_thres=args.nms_thres,verbose=True,rank=rank)

    elif rank==0:
        precision, recall, AP, f1, ap_class = evaluate_model_file(
            args.model,
            args.weights,
            valid_path,
            class_names,
            batch_size=args.batch_size,
            img_size=args.img_size,
            n_cpu=args.n_cpu,
            iou_thres=args.iou_thres,
            conf_thres=args.conf_thres,
            nms_thres=args.nms_thres,
            verbose=True,
            rank=rank)
    
    print("Total test time: %s seconds" % (time.time() - total_test_start_time))

# To obtain total testing time
total_test_start_time = time.time()

if __name__ == "__main__":
    run()
    

