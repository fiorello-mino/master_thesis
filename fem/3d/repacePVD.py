#!/usr/bin/env python3

import vtutools
import glob
import os
import sys
import re

try:
    filename = sys.argv[1]
except:
    filename = input("Please insert filename root: ")

list = glob.glob( filename + ".[0-9]*.pvtu" )
if len(list)==0:
    list = glob.glob( filename + "[0-9]*.vtu" )
    if len(list)==0:
        print ("No files")
        sys.exit()
    pvtu = False
    vtuExt = ".vtu"
else:
    pvtu = True
    vtuExt = ".pvtu"
list = sorted( list, key=lambda s: float(re.search(r'[+-]?[0-9]+\.[0-9]+', s).group()))

every = float(input("Time interval between files: "))
if every <= 0 :
    print ("invalid number")
    sys.exit()
    
print (os.getcwd())


keeplist = []
dellist = []

nextt=0
i=0
for it in range(0,len(list)):
    keep=False
    ifile = list[it]
    if pvtu:
        time = float(ifile.replace(filename + ".","").replace(".pvtu",""))
    else:
        time = float(ifile.replace(filename,"").replace(".vtu",""))

    if (time >= nextt-1.e-8 or i == len(list)-1 ):
        keep=True
        keeplist.append(ifile)
        while nextt-1.e-8 <= time:
            nextt += every
    elif(it < len(list)):
        ifile2 = list[it+1]
        if pvtu:
            time2 = float(ifile2.replace(filename + ".","").replace(".pvtu",""))
        else:
            time2 = float(ifile2.replace(filename,"").replace(".vtu",""))
 
        if (time2 > nextt+0.75*every):
            keep=True
            keeplist.append(ifile)
            while nextt-1.e-8 <= time:
                nextt += every
    if not keep: 
        dellist.append(ifile)
        if pvtu:
            timeTag = ifile.replace(filename + ".","").replace(".pvtu","")
            listvtu = glob.glob( filename + "-p[0-9]*-" + timeTag + ".vtu" )
            for vtufile in listvtu:
                dellist.append(vtufile)
    i += 1

vtkman = vtutools.VTK_XML_Serial_Unstructured()
for ifile in keeplist:
    if pvtu:
        vtkman.addtoPVD(filename, ifile.replace(filename + ".","").replace(".pvtu",""), vtuExt)
    else:
        vtkman.addtoPVD(filename, ifile.replace(filename,"").replace(".vtu",""), vtuExt)
vtkman.writePVD(filename + ".pvd")

if ( len(dellist) > 0 ):
    if ( (input("Really remove %i files (y/n)?" % len(dellist) )).lower() == "y" ):
        for ifile in dellist:
            os.remove(ifile)
        print ("Files removed!\n")
