import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt

import shutil
import os

from utils import *
from train import *
from eval import *

classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
               'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
               'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

def main(args):
    MODEL_NAME = f'{args.model}_{args.loss}_{args.scheduler}_{args.opt}' #

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

    model = get_model(args.model,args.untrained)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    counts = np.array(class_counts)
    criterion = get_loss(args.loss, counts, device)

    optimizer = get_optimizer(model.parameters(), args.opt, args.lr)

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
                       args.thresh)
        
    elif args.mode == 'eval':
        evaluate_model(device,
                       model,
                       dataloaders['test'],
                       criterion,
                       model_dir,
                       args.thresh)
        
    elif args.mode == 'pred':
        p_model_dir = os.path.join(args.save_dir, args.pred_model)
        p_image_dir = os.path.join(args.image_dir, args.pred_image)

        p_model_path = os.path.join(p_model_dir, 'best_model.pth')

        if not os.path.exists(p_model_path):
            raise FileNotFoundError(f"No model found at {p_model_path}. Please train the model before evaluation.")
        model.load_state_dict(torch.load(p_model_path))
        print('Model loaded succesfully and proceeding to eval')

        model.eval()

        transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(p_image_dir).convert("RGB")
        input_image = transform(image)
        input_image = input_image.unsqueeze(0)

        with torch.no_grad():
            output = model(input_image)

        output_probs = torch.sigmoid(output).cpu().numpy()
        binary_preds = (output_probs > 0.5).astype(int).flatten()

        pred = binary_preds.tolist()

        print(pred)

        model_dict = dict(
            type=args.model,
            arch=model.model,
            input_size=(256, 256)
        )

        gradcam = Grad_CAM(model_dict)

        for i in range(len(pred)):
            if pred[i] == 1:
                print(f'iter {i} with pred {pred[i]} and class {classes[i]}')
                mask, logit = gradcam(input_image, class_idx=i)

                heatmap, result = visualize_cam(mask, input_image)

                plt.imshow(result.permute(1, 2, 0).detach().numpy()) 
                plt.title(f'GradCAM for Class {classes[i]}')
                plt.axis('off')

                cam_dir = os.path.join(p_model_dir, 'gradcam')
                ensure_dir(cam_dir)
                cam_dir_s = os.path.join(cam_dir, f'prediction_{args.pred_image[:-4]}_{classes[i]}.png')

                plt.savefig(cam_dir_s)



    print('finish!')




if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, choices=['train','eval','pred'])
    parser.add_argument('--image_dir', default='./resized_images', type=str)
    parser.add_argument('--pred_model', default='res50_asl2_Adam', type=str)
    parser.add_argument('--pred_image', default='00000211_004.png', type=str)
    parser.add_argument('--train_csv', default='./labels/train_metadata.csv', type=str)
    parser.add_argument('--val_csv', default='./labels/val_metadata.csv', type=str)
    parser.add_argument('--test_csv', default='./labels/test_metadata.csv', type=str)
    parser.add_argument('--save_dir', default='./saves', type=str, help="directory where logs and model checkpoints will be saved")
    parser.add_argument('--model', default='res50', type=str, help="Neural Network model to be used", choices=['res18','res50','dense121','efficientb0','efficientb3'])
    parser.add_argument('--untrained', action='store_false', help="true if model is pretrained by ImageNet false if untrained")
    parser.add_argument('--thresh', action='store_true', help="Use dynamic threshold")
    parser.add_argument('--max_epochs', default=20, type=int, help="maximum number of epochs for training")
    parser.add_argument('--num_iter', default=None, type=int, help="maximum iterations taken per epoch")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size for data loaders")
    parser.add_argument('--img_size', default=256, type=int, help="desired size for image dataset")
    parser.add_argument('--loss', default='asl2', type=str, choices=['bce','bce_w','focal','asymmetric','asymmetric_avg','asl1','asl2','asl3'])
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--opt', default='Adam', type=str, choices=['SGD', 'SGD_Nesterov', 'Adamax','Adam','AdamW','RMSprop'])
    parser.add_argument('--scheduler', default='plateau1',type=str, choices=['plateau', 'plateau1', 'cyclic', 'cosine', 'warmupcosine'])
    parser.add_argument('--e_patience', default=10, type=int, help="patience the training has on epochs without learning before stopping")
    parser.add_argument('--s_patience', default=3, type=int, help="patience for the scheduler")
    args = parser.parse_args()

    print(args)

    main(args)