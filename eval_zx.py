import time
import torch
import subprocess
import os
import argparse
from model import EAST
from detect_zx import detect_dataset
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()
	
	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	os.chdir(submit_path)
	return
	res = subprocess.getoutput('zip -q submit_east_vgg16.zip *.txt')
	res = subprocess.getoutput('mv submit_east_vgg16.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit_east_vgg16.zip')
	print(res)
	# os.remove('./submit_east_vgg16.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', default='./pths/east_vgg16.pth')
	parser.add_argument('--test_img_path', default='./ICDAR_2015/test_img')
	parser.add_argument('--submit_path', default='./submit')
	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()

	args.test_img_path = os.path.abspath(args.test_img_path)
	eval_model(args.model_name, args.test_img_path, args.submit_path)
