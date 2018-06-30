###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import string

import torch
from torch.autograd import Variable

import data
import time
import sys
import math

import nltk.data # $ pip install http://www.nltk.org/nltk3-alpha/nltk-3.0a3.tar.gz
# python -c "import nltk; nltk.download('punkt')"

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

from random import randint
import random

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

parser.add_argument('--mint', type=float, default=0.65,
                    help='MINimum temperature')
parser.add_argument('--maxt', type=float, default=1.35,
                    help='MAXimum temperature')


args = parser.parse_args()



#############
# TEMPERATURE
#### RANGE ##

MIN_TEMP=args.mint#0.5
MAX_TEMP=args.maxt#1.0




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
# ppl= md.split("-")[9].split("_")[1]\n\nPoetry sources: a subset of Poetry Magazine, Jacket2, 2 River, Capa, Evergreen Review, Cathay by Li Bai, Kenneth Patchen, Maurice Blanchot, and previous Rerites.\nLyric sources: Bob Marley, Bob Dylan, David Bowie, Tom Waits, Patti Smith, Radiohead.\n\n+Tech-terminology source: jhavelikes.tumblr.com, 

det = "\tAveraged Stochastic Gradient Descent \n\twith Weight Dropped QRNN \n\tPoetry Generation \n\n\nTrained on 197,923 lines of poetry & pop lyrics. \n\n\nLibrary: PyTorch\nMode: QRNN\n\nEmbedding size: 400\nHidden Layers: 1550\nBatch size: 20\nEpoch: 478\nLoss: 3.62\nPerplexity: 37.16\n\nTemperature range: "+str(MIN_TEMP)+" to "+str(MAX_TEMP)


print("\n\n\n\n"+det)


#print("\nSystem will generate "+str(args.words)+" word bursts, perpetually, until stopped.")

#DISABLED   ########## print ("\nPress ANY key to get a new poem.\n")





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






######### SINCE straight input 
############### did NOT work on UBUNTU
################## this comoplex mess erupts

def read_single_keypress():
    """Waits for a single keypress on stdin.

    This is a silly function to call if you need to do it a lot because it has
    to store stdin's current setup, setup stdin for reading single keystrokes
    then read the single keystroke then revert stdin back after reading the
    keystroke.

    Returns the character of the key that was pressed (zero on
    KeyboardInterrupt which can happen when a signal gets handled)

    """
    import termios, fcntl, sys, os
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save) # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK 
                  | termios.ISTRIP | termios.INLCR | termios. IGNCR 
                  | termios.ICRNL | termios.IXON )
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios. PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1) # returns a single character
    except KeyboardInterrupt: 
        ret = 0
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret






#############################################
###########   INFINITE LOOP #################
#############################################



while(True):


    #SLEEP 
    #time.sleep(5)



   #print ("\n\n\n\n")
    
    torch.manual_seed(randint(0,9999999999))

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
            #print('                       {}/{} words'.format(i+1, args.words), end='\r')


        titl = words.split('\n', 1)[0].title()

        #erase the output '88/88 words' line
        #print('                                                                                                                       ', end='\r')

        words = words.replace(" \n","\n")        
        words = words.replace("\r","\n")
        words = words.replace("\n\n\n\n\n","\n\n")
        words = words.replace("\n\n\n\n","\n")
        words = words.replace("\n\n\n","\n")


        words = words.replace("\"","")
        words = words.replace("\“","")
        words = words.replace("\”","")
        
        words = words.replace(" \'","")
        words = words.replace("\' ","")


        words = words.replace("\’ ","")
        words = words.replace("\’","")

        words = words.replace(")","")
        words = words.replace("(","")
        words = words.replace("==="," ")

        words = words.replace("him-whose-penis-stretches-down-to-his-knees","")  
        words = words.replace(".the.cylinder-section.now.the.prism.cut.off.by.the.","")




        # TAKE AWAY ALL NATURAL line BREAKS
        words = words.replace("\n","")

    # NUMBER of lines
        minlines=60
        maxlines=60
        number_of_lines=20#randint(minlines,maxlines)

        w2="\n\n"
        li="\n\n\n\n\t"
        cnt=0
        


        # length of line
        maxl=72#randint(8,88)

        # FORMATTING SINGLE POEM ON SCREEN #
        for w in words.split(" "):
            
            li=li+" "+w

            

            if len(li)>maxl:


                cnt=cnt+1
                #TITLE
                if cnt==1:
                    li=li.strip()
                    if li[-1:]==",":
                            w2 = w2[:-1]
                    li+="\n"
                    li = "\n\t"+string.capwords(li)+"\n"
                else:
                    #VERSE
                    li=li.lstrip()
                    # li = li.capitalize()

                    sentences = sent_tokenizer.tokenize(li)
                    li = " ".join(sent.capitalize() for sent in sentences)

                    # EXIT VERSE
                if cnt> number_of_lines:
                    if w2[-1:]==",":
                        w2 = w2[:-1]
                    wp=w2+"\n\n"
                    #w2+="\n\t~+~"
                    break

                #spaces=""
                li="\t"+li
                li+="\n"
                w2+=li
                li=""#"\t     "
                

        words=w2.replace("~ + ~","    ")
        # for li in words.splitlines():
        #     if len(li)>maxl:
        #         words = "\t\n".join(words.splitlines()[1:])
        #         break







        ############# WAIT ##########
        ############# WAIT ##########
        ############# WAIT ##########
        ############# WAIT ##########
        ############# WAIT ##########
        ############# WAIT ##########
        ############# WAIT ##########
        #NOT ON LINUX input("Press Enter to continue...")
        #read_single_keypress()
       
        # SCREEN OUTPUT
        # for char in words:
        #     #time.sleep(0.01)
        #     sys.stdout.write(char)

        print(words)

        #print("\n\n\n\t\t\t\tTemperature= "+ str(math.ceil(args.temperature*100)/100)+"\tSeed:"+str(args.seed))

        #words+="\n\n\n\t\tTemperature="+ str(math.ceil(args.temperature*100)/100)+"\tSeed:"+str(args.seed)

        outf.write(wp)





outf.close()
