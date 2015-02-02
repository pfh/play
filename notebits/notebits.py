"""

execfile("/data/home/pfh/git/play/notebits/notebits.py")

"""

import IPython
ipython = IPython.get_ipython()

from IPython.display import display, HTML, Image

from rpy2 import robjects

import inspect, os


#Breaks postscript()
#ipython.magic("%reload_ext rmagic")

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

robjects.r['source']( os.path.join(path,"notebits.R") )

