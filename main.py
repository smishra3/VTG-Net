import numpy as np
import torch
import os
import csv
import math
from math import sqrt
import random
import cv2

print(torch.__version__)

from PIL import Image
from argparse import ArgumentParser
from skimage import transform
from scipy.stats.mstats import mquantiles
#from scipy.misc import imsave
#from scipy.misc import toimage
import torch.nn as nn

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage

from modules.dataset import NeuronEM2012
from modules.dataset import Glands
from modules.network import AUCU
from modules.criterion import CrossEntropyLoss2d
from modules.transform import Relabel, ToLabel, Colorize

from sklearn.metrics import jaccard_score

NUM_CHANNELS = 3
NUM_CLASSES = 2
cvParam = 0.95
ins = 192

output_feature = []


color_transform = Colorize()
image_transform = ToPILImage()
input_transform = Compose([
    ToTensor(),
])

target_transform = Compose([
    ToLabel(),
    Relabel(255, 1),
])



def get_gen(train_imgs, label_imgs):
    batch_sz = train_imgs.size(0)
    imageType = train_imgs.size(1)
    labelType = label_imgs.size(1)
    xs = train_imgs.size(2)
    ys = train_imgs.size(3)
    #print("tensor size in get_gen:")
    #print(batch_sz)
    #print(imageType)
    #print(xs)
    #print(ys)

    stx = random.randint(0, xs - ins)
    sty = random.randint(0, ys - ins)
    #print("generated random crop position:")
    #print(stx)
    #print(sty)

    train_samples = np.zeros((batch_sz, imageType, ins, ins))
    #print(train_imgs.size())
    label_samples = np.zeros((batch_sz, labelType, ins, ins))

    train_samples = train_imgs[:, :, stx:stx+ins, sty:sty+ins]
    label_samples = label_imgs[:, :, stx:stx+ins, sty:sty+ins]

    return train_samples, label_samples

# a simple custom collate function, just to show the idea
def my_collate(batch):
    print('batch start!')
    print(len(batch))
    #print(np.shape(batch))
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    print(type(images))
    #print(len(data))
    #print(len(target))
    #data = torch.cuda.DoubleTensor(data)
    #target = torch.cuda.LongTensor(target)
    return [images, labels]

def train(args, model):

    directory = os.path.dirname("checkpoint/")
    if not os.path.exists(directory):
        os.makedirs(directory)

   
    #weight = torch.ones(NUM_CLASSES)
    weight = torch.tensor([1/0.9105,1/0.0895]) #drive
    #weight = torch.tensor([1/0.9240,1/0.0760]) #stare
    #weight = torch.tensor([1/0.9213,1/0.0787]) #chasedb

    #loader = DataLoader( Glands(args.datadir, input_transform, target_transform, cvParam, True), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)
    #val_loader = DataLoader( Glands(args.datadir, input_transform, target_transform, cvParam, False), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)

    loader = DataLoader( NeuronEM2012(args.datadir, input_transform, target_transform, cvParam, True), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader( NeuronEM2012(args.datadir, input_transform, target_transform, cvParam, False), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    print('data:', len(loader))
    print('val:', len(val_loader))

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)
    optimizer = Adam(model.parameters(),lr = 0.00002)

    for epoch in range(1, args.num_epochs+1):

        model.train()

        epoch_loss = []
        val_epoch_loss = []
        for step, (images, labels) in enumerate(loader):
            print(images.shape)
            print(labels.shape)
            images, labels = get_gen(images, labels)
            #print(images.shape)
            #print(labels.shape)
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            inputs = Variable(images)
            targets = Variable(labels)
            look = targets[targets > 1]
            #print(len(targets)) 
            #print('Hi!')        
            
            outputs = model(inputs)
            #print(len(outputs))
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:,0])
            #loss1 = criterion(outputs, targets)
            #print(loss)
            #print(loss1)
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.data)
            
            if args.steps_loss > 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print("loss: {aver} (epoch: {epoch}, step: {step})".format(aver = average, epoch = epoch, step = step))
            if (epoch > 1000 and epoch % args.epoch_save == 0 and step ==0) or (epoch == 2 and step == 0):
                filename ="checkpoint/" + "{model}-{epoch:03}-{step:04}.pth".format(model = args.model, epoch = epoch, step = step)
                torch.save(model.state_dict(), filename)
                evaluate(args, model,epoch)
            if (epoch % 50 == 0 and step ==0):
                evaluate(args, model,epoch)

        if len(val_loader) > 0:
            for step, (images, labels) in enumerate(val_loader):
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                inputs = Variable(images)
                targets = Variable(labels)
                
                outputs = model(inputs)
                #print("Good!")

                val_loss = criterion(outputs, targets[:,0])

                val_epoch_loss.append(val_loss.data)
                
                if args.steps_loss > 0 and step % args.steps_loss == 0:
                   val_average = sum(val_epoch_loss) / len(val_epoch_loss)

        
        with open("loss.csv", "a") as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow([epoch, average])


def evaluate(args, model, epoch):

    direc = args.output_dir + format(epoch,'04') + '/'
    directory = os.path.dirname(direc)

    if not os.path.exists(directory):
        os.makedirs(directory)

    model.eval()
    softmax = nn.Softmax2d()
    softmax.cuda()

    avk = 4
    nrotate =4

    filenames = [os.path.basename(os.path.splitext(f)[0])
        for f in os.listdir(args.input_dir) if f.endswith(args.data_type)]
    filenames.sort()

    #print('####################')
    #print('Here!!!!!!!!!!!!!!!!!!!!!')
    #print(filenames)
    #print(xyz)

    for fn in filenames:
        full_fn = "{file}{ext}".format(file = fn, ext = args.data_type)
        print(full_fn)
       
        #full_ln = "{file}{ext}".format(file = (fn + '_segmentation'), ext = args.data_type)
        #full_ln = "{file}{ext}".format(file = (fn + '_segmentation'), ext = '.png')
        #print(full_ln)
 
        with open(os.path.join(args.input_dir, full_fn),'rb') as f:
            image = Image.open(f).convert('RGB')

        image = image.resize((512,512), resample=Image.BICUBIC)
        #with open(os.path.join(args.mask_dir, full_ln),'rb') as f:
        #    mask = Image.open(f).convert('1')        

        print("before stiching:")
        print(image.size)

        wI = np.zeros([ins, ins])
        pmap = np.zeros([image.size[1], image.size[0]])
        avI = np.zeros([image.size[1], image.size[0]])
        #print('size: ', image.size[1], image.size[0])
        for i in range(ins):
            for j in range(ins):
                dx=min(i,ins-1-i)
                dy=min(j,ins-1-j)
                d=min(dx,dy)+1
                wI[i,j]=d
        wI = wI/wI.max()
        #print(wI.min())
        #print(wI)

        for i1 in range(math.ceil(float(avk)*(float(image.size[0])-float(ins))/float(ins))+1):
            for j1 in range(math.ceil(float(avk)*(float(image.size[1])-float(ins))/float(ins))+1):
                insti=math.floor(float(i1)*float(ins)/float(avk))
                instj=math.floor(float(j1)*float(ins)/float(avk))
                inedi=insti+ins
                inedj=instj+ins
                if inedi>image.size[0]:
                    inedi=image.size[0]
                    insti=inedi-ins
                if inedj>image.size[1]:
                    inedj=image.size[1]
                    instj=inedj-ins
                #print(insti,inedi,instj,inedj)
                #print(insti,instj,inedi,inedj)

                small_pmap = np.zeros([ins, ins])
                for i in range(nrotate):
                    small_in = image.crop((insti,instj,inedi,inedj))
                    small_in = small_in.rotate(90*i)

                    tI = input_transform(small_in)
                    label = model(Variable(tI.cuda(), volatile=True).unsqueeze(0))
                    prob = softmax(label)
                    xx = prob.cpu().data.float().select(1,1)
                    small_out = image_transform(xx)

                    small_out = small_out.rotate(-90*i)

                    small_pmap = small_pmap + np.array(small_out)

                small_pmap = small_pmap/nrotate
                #print('small: ', small_pmap.size)
                #print('wI: ', wI.size)
                pmap[instj:inedj, insti:inedi] += np.multiply(small_pmap, wI)
                avI[instj:inedj, insti:inedi] += wI
        xx = np.divide(pmap, avI)
        #print(xx.size)
        output = Image.fromarray(xx).convert('L')
        
        out_str = args.output_dir + format(epoch,'04') + '/' + fn + '.png'      
        
        output.save(out_str)

def main(args):

    model = None
    if args.model == 'aucu':
        model = AUCU(NUM_CLASSES, NUM_CHANNELS)
        print('aucu')

    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    if args.state:
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))

    print(model)

    if args.mode == 'eval':
        evaluate(args, model, 200)
        print('success!')
    if args.mode == 'train':
        train(args, model)
   
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--model', required=True)
    parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('--input_dir')
    parser_eval.add_argument('--output_dir')
    parser_eval.add_argument('--data_type') 

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--port', type=int, default=80)
    parser_train.add_argument('--datadir', required=True)
    parser_train.add_argument('--input_dir')
    parser_train.add_argument('--output_dir')
    parser_train.add_argument('--mask_dir')
    parser_train.add_argument('--data_type')
    parser_train.add_argument('--num-epochs', type=int, default=5001)
    parser_train.add_argument('--num-workers', type=int, default=1)
    parser_train.add_argument('--batch-size', type=int, default=1)
    parser_train.add_argument('--steps-loss', type=int, default=50)
    parser_train.add_argument('--steps-plot', type=int, default=0)
    parser_train.add_argument('--epoch_save', type=int, default=200)

    main(parser.parse_args())
