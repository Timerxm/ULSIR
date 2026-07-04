import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from tqdm import tqdm  
import torch.nn.functional as F
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
import cv2
from data.select_dataset import define_Dataset
from models.select_model import define_Model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main(json_path='options/train_msrresnet_psnr.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/swinir/weixing.json', help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = False

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        torch.cuda.set_device(args.local_rank)
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    epoch = max(init_iter_G, init_iter_E, init_iter_optimizerG) + 1

    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters per epoch: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=0,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    model = define_Model(opt)
    model.init_train()
    best_psnr = -1.0
    best_model_path = None

    max_epochs = 10000

    for epoch in range(epoch, 100000):
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)

        # Training loss record for each epoch.
        epoch_loss = 0.0
        
        # Show only the batch-level progress bar, including the current epoch.
        if opt['rank'] == 0:
            train_iterator = tqdm(enumerate(train_loader), total=len(train_loader),
                                 desc=f"Epoch {epoch}/{max_epochs}", unit="batch", leave=True)
        else:
            train_iterator = enumerate(train_loader)
            
        # Training iteration with the batch progress bar.
        for i, train_data in train_iterator:
         
            # Feed data.
            model.feed_data(train_data)
            
            # Optimize parameters.
            model.optimize_parameters(epoch)         
            
            # Accumulate loss.
            current_loss = model.current_log()['total_loss']
            epoch_loss += current_loss
            
            # Show the current loss and learning rate in the progress bar.
            if opt['rank'] == 0:
                train_iterator.set_postfix({
                    "Loss": f"{current_loss:.3e}",
                    "LR": f"{model.current_learning_rate():.3e}"
                })

        # Update the learning rate.
        model.update_learning_rate()

        # Compute the average loss for this epoch.
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # Print training information at the end of the epoch.
        if opt['rank'] == 0:
            logger.info('<epoch:{:3d}, average loss: {:.3e}, lr:{:.3e}>'.format(
                epoch, avg_epoch_loss, model.current_learning_rate()))

        # Run testing and compute PSNR at the end of the epoch, without a progress bar.
        if opt['rank'] == 0:
            avg_psnr = 0.0
            idx = 0

            for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # Save the estimated image.
                    save_img_path = os.path.join(img_dir, '{:s}_epoch{:d}.png'.format(img_name, epoch))
                    util.imsave(E_img, save_img_path)

                    # Compute and print PSNR.
                    current_psnr = util.calculate_psnr(E_img, H_img)
                    avg_psnr += current_psnr
                    #print(f'Testing {idx}/{len(test_loader)}: {image_name_ext} - PSNR: {current_psnr:.2f}dB')

            avg_psnr = avg_psnr / idx
            logger.info('<epoch:{:3d}, Average PSNR : {:<.2f}dB\n'.format(epoch, avg_psnr))
            print(f'Epoch {epoch} - Average PSNR: {avg_psnr:.2f}dB')

            if avg_psnr > best_psnr:
                    print(f'PSNR improved from {best_psnr:.2f}dB to {avg_psnr:.2f}dB, saving model...')
                    best_psnr = avg_psnr
                    
                    current_model_path = model.save(epoch)
                    
                    if best_model_path and os.path.exists(best_model_path):
                        try:
                            os.remove(best_model_path)
                            logger.info(f'Removed previous best model: {best_model_path}')
                            print(f'Removed previous best model: {best_model_path}')
                        except Exception as e:
                            logger.warning(f'Failed to remove previous model: {e}')
                            print(f'Warning: Failed to remove previous model: {e}')
                    
                    best_model_path = current_model_path
            else:
                    logger.info(f'PSNR did not improve (current best: {best_psnr:.2f}dB)')
                    print(f'PSNR did not improve (current best: {best_psnr:.2f}dB)')

    logger.info('Training completed. Best PSNR: {:.2f}dB'.format(best_psnr))
    print(f'Training completed. Best PSNR: {best_psnr:.2f}dB')

if __name__ == '__main__':
    main()
    
