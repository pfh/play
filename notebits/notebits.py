"""

%load_ext rmagic
execfile("/data/home/pfh/git/play/notebits/notebits.py")

"""

import IPython
ipython = IPython.get_ipython()

#Breaks postscript()
ipython.magic("%reload_ext rmagic")

from IPython.display import display, HTML, Image

from rpy2 import robjects

import inspect, os


filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

robjects.r['source']( os.path.join(path,"notebits.R") )

print "Loaded utility functions from https://github.com/pfh/play/tree/master/notebits"