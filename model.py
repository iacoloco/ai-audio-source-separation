#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 18:51:36 2026

@author: armandoiachini
"""
import torch.nn as nn


class MLP_Class(nn.Module):
    def __init__(self, n_freq ):
        super().__init__()
        
        self.layer1 = nn.Linear(n_freq, 512)
        self.activation1 = nn.ReLU()
        
        self.layer2 = nn.Linear(512 , 256)
        self.activation2 = nn.ReLU()
        
        self.layer3= nn.Linear( 256 , 512 )
        self.activation3 = nn.ReLU()
        
        self.layer4 = nn.Linear(512 , n_freq)    
        self.out = nn.Sigmoid()
        
    def forward(self, x):
            x = self.activation1(self.layer1(x))
            x = self.activation2(self.layer2(x))
            x = self.activation3(self.layer3(x))
            x = self.out(self.layer4(x))
            return x
    
    
        
        