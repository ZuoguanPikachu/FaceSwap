import os
import cv2
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader
from models.swapnet import SwapNet128
from dataset.face_pair_dataset import FacePairDataset128x128
from torchvision import transforms
import time


batch_size = 2
epochs = 100000
save_per_epoch = 200
log_per_epoch = 100

a_dir = 'face/MilaAzul'
b_dir = 'face/Dlrb'
log_img_dir = 'checkpoint/results'
check_point_save_path = "checkpoint/FaceSwap128.pth"



def main():
    os.makedirs(log_img_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform = transforms.Compose([transforms.RandomHorizontalFlip()])
    ds = FacePairDataset128x128(a_dir=a_dir, b_dir=b_dir, transform=transform)
    dataloader = DataLoader(ds, batch_size, shuffle=True)

    model = SwapNet128()
    model.to(device)
    start_epoch = 0
    print('try resume from checkpoint')
    if os.path.isdir('checkpoint'):
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(check_point_save_path)
            else:
                checkpoint = torch.load(
                    check_point_save_path, map_location={'cuda:0': 'cpu'})
            model.load_state_dict(checkpoint['state'])
            start_epoch = checkpoint['epoch']
            print('checkpoint loaded.')
        except FileNotFoundError:
            print('Can\'t found FaceSwap128.pth')

    criterion = nn.L1Loss()
    optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_A.parameters()}], lr=5e-5, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_B.parameters()}], lr=5e-5, betas=(0.5, 0.999))

    print('Start training, from epoch {} '.format(start_epoch))
    try:
        for epoch in range(start_epoch, epochs):
            iter = 0
            iter_count = len(dataloader)
            for data in dataloader: 
                iter += 1
                img_a_target, img_a_input, img_b_target, img_b_input = data
                img_a_target = img_a_target.to(device)
                img_a_input = img_a_input.to(device)
                img_b_target = img_b_target.to(device)
                img_b_input = img_b_input.to(device)

                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                predict_a = model(img_a_input, select='A')
                predict_b = model(img_b_input, select='B')
                loss1 = criterion(predict_a, img_a_target)
                loss2 = criterion(predict_b, img_b_target)
                loss1.backward()
                loss2.backward()
                optimizer_1.step()
                optimizer_2.step()

                if epoch % log_per_epoch == 0:
                    if iter == 1:
                        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

                    print('Epoch: {}, iter: {}, lossA: {:.8f}, lossB: {:.8f}'.format(str(epoch).zfill(6), iter, loss1.item(), loss2.item()))

                    if iter == iter_count:
                        img_a_original = np.array(img_a_target.detach().cpu().numpy()[0].transpose(1, 2, 0)*255, dtype=np.uint8)

                        a_predict_b = model(img_a_input, select='B')
                        a_predict_b = np.array(a_predict_b.detach().cpu().numpy()[0].transpose(1, 2, 0)*255, dtype=np.uint8)

                        cv2.imwrite(os.path.join(log_img_dir, '{}_0.png'.format(str(epoch).zfill(6))), cv2.cvtColor(img_a_original, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(log_img_dir, '{}_2.png'.format(str(epoch).zfill(6))), cv2.cvtColor(a_predict_b, cv2.COLOR_BGR2RGB))
                        print('Record a result')

                if epoch % save_per_epoch == 0 and iter==iter_count:
                    print('Saving models...')
                    state = {
                        'state': model.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(state, check_point_save_path)

    except KeyboardInterrupt:
        print('try saving models...do not interrupt')
        state = {
            'state': model.state_dict(),
            'epoch': epoch
        }
        torch.save(state, check_point_save_path)


if __name__ == "__main__":
    main()
