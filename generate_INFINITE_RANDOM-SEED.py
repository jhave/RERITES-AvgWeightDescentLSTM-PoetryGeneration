###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data
import time
import sys
import math


from random import randint
import random

#############
# TEMPERATURE
#### RANGE ##

MIN_TEMP=0.5
MAX_TEMP=1.0


parser = argparse.ArgumentParser(description='PyTorch Poetry Language Model')

from random import randint
from datetime import datetime
started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())




# Model parameters.
parser.add_argument('--data', type=str, default='./data/pf',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='GENERATED/generated-'+ started_datestring +'.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()





##########################################
############   DISPLAY ###################
##########################################


# GET TECH DETAILS
# md=args.checkpoint.split("/")[-1]
# style = md.split("-")[1]
# emsize= md.split("-")[3]

# nhid= md.split("-")[4].split("_")[1]
# nlay= md.split("-")[5].split("_")[1]
# bs = md.split("-")[6].split("_")[2]
# ep= md.split("-")[7].split("_")[1]
# loss= md.split("-")[8].split("_")[1]
# ppl= md.split("-")[9].split("_")[1]

det = "\tAveraged Stochastic Gradient Descent \n\twith Weight Dropped QRNN \n\tPoetry Generation \n\n\nTrained on 197,923 lines of poetry & pop lyrics. \n\nPoetry sources: a subset of Poetry Magazine, Jacket2, 2 River, Capa, Evergreen Review, Cathay by Li Bai, Kenneth Patchen, Maurice Blanchot, and previous Rerites.\nLyric sources: Bob Marley, Bob Dylan, David Bowie, Tom Waits, Patti Smith, Radiohead.\n\n+Tech-terminology source: jhavelikes.tumblr.com, \n\n\n+~+Library: PyTorch (word-language-model modified by Salesforce Research)+~+\n\nMode: QRNN\nEmbedding size: 400\nHidden Layers: 1550\nBatch size: 20\nEpoch: 478\nLoss: 3.62\nPerplexity: 37.16\n\nTemperature range: "+str(MIN_TEMP)+" to "+str(MAX_TEMP)


print("\n\n\n\n"+det)


print("\n\nSystem will generate "+str(args.words)+" word bursts, perpetually, until stopped.")

print ("\nInitializing.Please be patient.\n\nSYSTEM OUTPUT : REAL-TIME generation on TitanX GPU \nre-loading model every poem \nfresh with a new RANDOM SEED.\n")








#############################################
###########   INFINITE LOOP #################
#############################################



while(True):

    # Set the random seed RANDOMLY for UNreproducibility.
    args.seed=randint(0,9999999999)


    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f)
    model.eval()
    if args.model == 'QRNN':
        model.reset()

    if args.cuda:
        model.cuda()
    else:
        model.cpu()

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
    if args.cuda:
        input.data = input.data.cuda()





    # SLEEP time.sleep(5)



    print ("\n\n\n\t\t~ + ~\n\n")
    
    #torch.manual_seed(randint(0,9999999999))

    words=''

    ###########################################
    ######### RANDOM TEMPERATURE ##############
    args.temperature = random.uniform(MIN_TEMP, MAX_TEMP)



    with open(args.outf, 'a') as outf:
        for i in range(args.words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx)


            if word_idx<=len(corpus.dictionary.idx2word)-1:
                word = corpus.dictionary.idx2word[word_idx]

                if word == '<eos>':
                    word = '\n'

                if word == '&amp;':
                    word = '\n'

                words+=word+" "

            #outf.write(word + ('\n' if i % 20 == 19 else ' '))

               # Output how many created so far
            print('                       {}/{} words'.format(i+1, args.words), end='\r')


        titl = words.split('\n', 1)[0].title()

        #erase the output '88/88 words' line
        print('                                                                                                                       ', end='\r')

        maxl=75
        for li in words.splitlines():
            if len(li)>maxl:
                words = "\n".join(words.splitlines()[1:])
                break

        words = words.replace(" \n","\n")        
        words = words.replace("\r","\n")
        words = words.replace("\n\n\n\n\n","\n\n")
        words = words.replace("\n\n\n\n","\n")
        words = words.replace("\n\n\n","\n")

       
        # SCREEN OUTPUT
        for char in words:
            #time.sleep(0.01)
            sys.stdout.write(char)

        print("\n\n\n\t\t\t\tTemperature= "+ str(math.ceil(args.temperature*100)/100)+"\tSeed:"+str(args.seed))

        words+="\n\n\n\t\tTemperature="+ str(math.ceil(args.temperature*100)/100)+"\tSeed:"+str(args.seed)+"\n\n\n\t\t\t\t~ + ~\n\n\n"#--------------------------------------------------------------------------------------------\nGenerated on : "+str(started_datestring)+"\n--------------------------------------------------------------------------------------------\n\nTech details\n-------------  \n\nInfo: http://bdp.glia.ca/\nCode: https://github.com/jhave/pytorch-poetry-generation\n\n"+det+"\n\n--------------------------------------------------------------------------------------------\n"+args.checkpoint
        outf.write(words)

outf.close()