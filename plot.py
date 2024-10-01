import os
import pandas as pd
import matplotlib.pyplot as plt

import argparse

def plot_history(args):
    data_frames = {}

    vars = {
        'epoch': 0,
        'val_loss': 1,
        'val_auc': 2,
        'precision': 3,
        'recall': 4,
        'f1_score': 5,
        'lr': 6,
        'train_time': 7,
        'val_time': 8,
        'epoch_time': 9,
    }

    if args.target_var not in vars:
        raise ValueError(f"Invalid selected variable: {args.target_var}. Valid options are: {list(vars.keys())}")
    

    for subfolder in os.listdir(args.saves_dir):
        subfolder_path = os.path.join(args.saves_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            csv_path = os.path.join(subfolder_path, 'history.csv')
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path, header=None)
                
                # Add the data frame to the dictionary
                data_frames[subfolder] = df

    # Plotting
    plt.figure(figsize=(12, 6))
    
    for subfolder, df in data_frames.items():
        plt.plot(df[vars['epoch']], df[vars[args.target_var]], label=subfolder)

    
    # Customize the plot
    plt.title(f'Plot of {args.target_var} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(args.target_var)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Show or save the plot
    plt.savefig(os.path.join(args.output_dir, args.name))

if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--saves_dir', default='./saves', type=str)
    parser.add_argument('--output_dir', default='./docu', type=str)
    parser.add_argument('--target_var', default='val_auc', type=str, help="Variable to plot")
    parser.add_argument('--name', default='loss_selection_plot.png', type=str)

    args = parser.parse_args()

    print(args)

    plot_history(args)
