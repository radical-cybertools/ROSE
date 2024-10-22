import os

import torch

import trainers


def load_trainer(device, std_tr, s):
    return trainers.DeepDiffusionTrainer(device=device, std_tr=std_tr, s=s)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    save_loc = r'C:\deep_diff_test'
    if not os.path.isdir(save_loc):
        os.mkdir(save_loc)

    load_loc = r'C:\folder_with_data_sets'

    trainer = load_trainer(device=device, std_tr=0.01, s=512)

    BATCH_SIZE = 25
    NUM_WORKERS = 4
    lr = 0.0001
    dataset_name = '1SourcesRdm'
    error_list, test_error, test_error_plus, test_error_minus, save_loc = \
        trainer.train_diff_solver(load_loc, save_loc, lr, BATCH_SIZE, NUM_WORKERS,
                                  dataset_name=dataset_name, transforemation='log')
