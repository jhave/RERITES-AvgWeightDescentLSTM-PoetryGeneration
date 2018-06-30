import random
import time
import sys


#!/usr/bin/env python
import curses
 
# get the curses screen window
screen = curses.initscr()
 
# turn off input echoing
curses.noecho()
 
# respond to keys immediately (don't wait for enter)
curses.cbreak()
 
# map arrow keys to special values
screen.keypad(True)




text_file = open("ohhla_RESERVOIR.txt", "r")
txt = text_file.read()

#  # TAKE AWAY ALL NATURAL line BREAKS
txt = txt.replace("\n"," ")
txt = txt.replace("\r"," ")
txt = txt.replace("\t"," ")
txt = txt.replace("  "," ")


words= txt.split(" ")




def main(win):
    win.nodelay(True) # make getkey() not wait
    x = random.randint(0,len(words)-1)



    delayed=0.000001
    cnt=0

    while True:

        cnt=cnt+1

        #PRINT
        sys.stdout.write(words[x]+" ")
        time.sleep(delayed)

       # next word
        if cnt>5000:
            cnt=0
            x += 1
            # sys.stdout.write(" * * * ")
            if x>=len(words):
                x=0

        
        try:
            char = screen.getch()
        except: # in no delay mode getkey raise and exeption if no key is press 
            char = None
        if char == ord(' '): # space then break
            break
        elif char == curses.KEY_RIGHT:
            # next word
            x += 1
            sys.stdout.write(" * * * ")
            if x>=len(words):
                x=0
        elif char == curses.KEY_LEFT:
              # next word
            x -= 1
            sys.stdout.write(" * * * ")
            if x==0:
                x=len(words)-1     
        elif char == curses.KEY_UP:
            delayed=delayed+0.000001      
        elif char == curses.KEY_DOWN:
            delayed=delayed-0.000001 
            #sys.stdout.write("  DELAYED::::::::::::::::::::"+str(delayed))
            if delayed<=0:
                delayed=0
            #sys.stdout.write(" AFTER DELAYED::::::::::::::::::::"+str(delayed))
            

#a wrapper to create a window, and clean up at the end
curses.wrapper(main)