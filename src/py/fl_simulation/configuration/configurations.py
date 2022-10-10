import logging
from sys import stdout
from os import getcwd

class Configurations():
    def __init__(self):
        # Setup Logging 
        self.root = logging.getLogger()
        self.root.setLevel(logging.DEBUG)

        if (self.root.hasHandlers()):
            self.root.handlers.clear()
        handler = logging.StreamHandler(stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.root.addHandler(handler)
        
        # working directory
        self.cwd = getcwd()
