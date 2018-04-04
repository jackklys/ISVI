from __future__ import print_function
from data import *
from engine import *
import cPickle as pkl
import math
import sys
import os.path
import seaborn as sns

def plot_loss(loss_curve):
    plt.plot(loss_curve)
    plt.xlabel('epochs')
    plt.ylabel('negative log-likelihood')

def plot_grads(grads):
    colors = ['red', 'blue', 'green']
    for i, x in enumerate(grads):
        for y in x:
            plt.plot(y, color=colors[i])
    plt.xlabel('epochs')
    plt.ylabel('gradient')

def plot_gradvars(gradvars):
    colors = ['red', 'blue', 'green']
    for i, x in enumerate(gradvars):
        for y in x:
            plt.plot(y, color=colors[i])
    plt.xlabel('epochs')
    plt.ylabel('variance of gradient')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-k', type=int, default=12)
	parser.add_argument('-d', type=int, default=2)
	parser.add_argument('-target', type=str, default='HalfGaussian')
	args = parser.parse_args()
	target = args.target
	k = args.k

	d = args.d
	print('{}d'.format(d))

	save_dir = 'results/{}d/{}/{}/'.format(d, k, target)

	sns.set_style('darkgrid')
	sns.despine(left=True, bottom=True)

	all_models = ['IWAE', 'IWAEN', 'IWAE2', 'COVR4', 'COVR6', 'COVK2', 'COVK', 'COVD', 'COVF', 'COVF2', 'COVKA', 'COVKB']
	models = []

	for m in all_models:
		if os.path.exists(save_dir + m):
			models.append(m)

	# for i in range(100):
	# 	losses_dir =  save_dir + 'loss{}.txt'.format(i)
	# 	if not os.path.exists(losses_dir):
	# 		break

	for i, m in enumerate(models):
		with open(save_dir + m + '/loss.pkl', 'rb' ) as f:
			curves = pkl.load(f)

		print(m)
		f = plt.figure(1)
		sns.set_context(rc={"lines.linewidth": 0.2})
		ax = f.add_subplot(len(models),3,3*i+1)
		ax.set_title(m, rotation='vertical',x=-0.2,y=0.5)
		plot_loss(curves[0])
		# ax.annotate(str(curves[0][-1]), xy=(len(curves[0]), curves[0][-1]), xytext=(-1,0))
		ax.text((4./5)*len(curves[0]), max(curves[0]), '{:.5f}'.format(curves[0][-1]), fontsize=6)

		f.add_subplot(len(models),3,3*i+2)
		plot_grads(curves[1])

		f.add_subplot(len(models),3,3*i+2)
		plt.subplot(len(models),3,3*i+3)
		plot_gradvars(curves[2])
		# print(curves[2][-1])

		plt.figure(2)
		sns.set_context(rc={"lines.linewidth": 0.8})
		plt.plot(curves[0], label=m)
		print(curves[0][-1])

	matplotlib.rcParams.update({'font.size': 4})

	plt.figure(1)
	dpi = plt.figure(1).get_dpi()
	dpi = 2*dpi
	for i in range(100):
		d = save_dir + 'gradients{}.png'.format(i)
		if not os.path.exists(d):
			plt.savefig(d, dpi= dpi)
			plt.close()
			break


	matplotlib.rcParams.update({'font.size': 8})

	sns.set_context(rc={"lines.linewidth": 1.5})
	plt.figure(2)
	dpi = plt.figure(2).get_dpi()
	plt.xlabel('epochs')
	plt.ylabel('negative log-likelihood')
	plt.legend()
	plt.savefig(save_dir + 'loss.png', dpi= 1.5*dpi)
	plt.close()

