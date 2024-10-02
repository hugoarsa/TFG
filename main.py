import argparse
import numpy as np
import pandas as pd
import torch
import torchvision

import shutil
import os

from utils import *
from train import *
from eval import *



def main(args):
    MODEL_NAME = f'shedulerexp_{args.model_name}_{args.loss}_{args.scheduler}_{args.opt}'

    ensure_dir(args.save_dir)

    model_dir = os.path.join(args.save_dir, MODEL_NAME)

    ensure_dir(model_dir)

    # Potentially setting seeds (not now)

    dataloaders, dataset_sizes, class_counts = make_data_loaders(args.train_csv, 
                                                                 args.val_csv,
                                                                 args.test_csv, 
                                                                 args.image_dir, 
                                                                 args.batch_size, 
                                                                 args.img_size)

    model = get_model(args.model_name,args.pretrained)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    counts = np.array(class_counts)
    criterion = get_loss(args.loss, counts, device)

    optimizer = get_optimizer(model.parameters(), args.opt, args.lr)

    print(args.scheduler)

    scheduler = get_scheduler(optimizer,args.scheduler)

    if args.mode == 'train':

        model = train_model(device, 
                            model, 
                            model_dir,
                            dataloaders['train'], 
                            dataloaders['val'],
                            criterion,
                            optimizer,
                            scheduler,
                            args.max_epochs,
                            args.num_iter,
                            args.s_patience)
        
        evaluate_model(device,
                       model,
                       dataloaders['test'],
                       criterion,
                       model_dir,
                       False)
        
    elif args.mode == 'eval':
        evaluate_model(device,
                       model,
                       dataloaders['test'],
                       criterion,
                       model_dir,
                       False)


    # elif args.mode == 'pred'

    print('finish!')




if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, choices=['train','eval','pred'])
    parser.add_argument('--image_dir', default='./resized_images', type=str)
    parser.add_argument('--train_csv', default='./labels/train_metadata.csv', type=str)
    parser.add_argument('--val_csv', default='./labels/val_metadata.csv', type=str)
    parser.add_argument('--test_csv', default='./labels/test_metadata.csv', type=str)
    parser.add_argument('--save_dir', default='./saves', type=str, help="directory where logs and model checkpoints will be saved")
    parser.add_argument('--model_name', default='res18', type=str, help="Neural Network model to be used", choices=['res18','res50','dense121','efficientb0','efficientb3'])
    parser.add_argument('--pretrained', default=True, type=bool, help="true if model is pretrained by ImageNet")
    parser.add_argument('--thresh', default=False, type=bool, help="true eval is to use dynamic threshold")
    parser.add_argument('--max_epochs', default=20, type=int, help="maximum number of epochs for training")
    parser.add_argument('--num_iter', default=None, type=int, help="maximum iterations taken per epoch")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size for data loaders")
    parser.add_argument('--img_size', default=256, type=int, help="desired size for image dataset")
    parser.add_argument('--loss', default='asl2', type=str, choices=['bce','bce_w','focal','asymmetric','asymmetric_avg','asl1','asl2','asl3'])
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--opt', default='Adam', type=str, choices=['SGD', 'SGD_Nesterov', 'Adadelta','Adam','AdamW','RMSprop'])
    parser.add_argument('--scheduler', default='plateau',type=str, choices=['plateau', 'plateau1', 'cyclic', 'cosine', 'warmupcosine'])
    parser.add_argument('--e_patience', default=10, type=int, help="patience the training has on epochs without learning before stopping")
    parser.add_argument('--s_patience', default=3, type=int, help="patience for the scheduler")
    args = parser.parse_args()

    print(args)

    main(args)