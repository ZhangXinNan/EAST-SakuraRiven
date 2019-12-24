#encoding=utf-8
import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
# from dataset import custom_dataset
# from model import EAST
# from loss import Loss
from dataset_md_screen import custom_dataset_md_screen
from model_md_screen import EAST_md_screen as EAST
from loss_md_screen import Loss_md_screen as Loss
import os
import argparse
import time
import numpy as np


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, pretrained=None, length=512):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset_md_screen(train_img_path, train_gt_path, length=length)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(length=length)
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	if pretrained is not None:
		model.load_state_dict(torch.load(pretrained))
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	epoch_loss_list = []
	for epoch in range(epoch_iter):
		model.train()
		scheduler.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			# loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			loss = criterion(gt_score, pred_score, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))

		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))
		epoch_loss_list.append(epoch_loss)
		print(epoch_loss_list)
	return epoch_loss_list

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_img', default=os.path.abspath('D:\\data_md\\liuchenxing\\20191120_screen\\train_img'))
	parser.add_argument('--train_gt', default=os.path.abspath('D:\\data_md\\liuchenxing\\20191120_screen\\train_gt'))
	parser.add_argument('--pths_path', default='./pths_zx_md_screen512_')
	parser.add_argument('--pretrained_model', default=None)
	parser.add_argument('--learn_rate', default=1e-3, type=float)
	parser.add_argument('--length', default=512, type=int)
	parser.add_argument('--batch_size', default=8, type=int)
	return parser.parse_args()


def main(args):
	pths_path = args.pths_path
	if args.pretrained_model is not None and not os.path.isfile(args.pretrained_model):
		pretrained_model = None
	else:
		pretrained_model = args.pretrained_model

	if not os.path.isdir(pths_path):
		os.makedirs(pths_path)
	# 超过8就会出现显存不足
	# batch_size = 8
	lr = 1e-3 if args.learn_rate is None else args.learn_rate
	# num_workers    = 4
	# TypeError: function takes exactly 5 arguments (1 given)
	# 改为0，可以跳过此错误
	num_workers = 0
	epoch_iter = 100
	save_interval = 5
	epoch_loss = train(args.train_img, args.train_gt, pths_path, args.batch_size, lr, num_workers, epoch_iter,
					   save_interval, pretrained_model, args.length)
	print(epoch_loss)


if __name__ == '__main__':
	main(get_args())

