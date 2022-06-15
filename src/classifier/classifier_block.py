from collections import OrderedDict
import torch.nn as nn
import torch

class ClassifierHead(nn.Module):
    def __init__(self, in_c, num_classes, channels=None, dropout=.5, ensemble=False, sub=False):
        super().__init__()
        self.c = [in_c, 512, 256, 128, 64]
        if sub:
            self.c = [in_c, in_c, 512, 256, 128, 64]
        if channels is not None:
            self.c = channels
        self.d = dropout
        self.pre_head_dim = self.c[-1]
        self.ensemble = ensemble
        self.sub = sub
        
        if ensemble:
            self.pre_dense_1 = self.create_dense(
                fin=self.c[0],
                fout=self.c[1],
                d=self.d
            )
            self.pre_dense_2 = self.create_dense(
                fin=1024,
                fout=self.c[1],
                d=self.d
            )
            self.denses = nn.Sequential(
                OrderedDict(
                    [
                        (
                            f'dense_block_{i}', 
                            self.create_dense(
                                fin=self.c[i],
                                fout=self.c[i+1],
                                d=self.d
                            )
                        ) for i in range(1, len(self.c)-1)
                    ]
                )
            )
        elif sub:
            self.pre_dense = self.create_dense(
                fin=self.c[0],
                fout=self.c[1]//2,
                d=self.d
            )
            self.denses = nn.Sequential(
                OrderedDict(
                    [
                        (
                            f'dense_block_{i}', 
                            self.create_dense(
                                fin=self.c[i],
                                fout=self.c[i+1],
                                d=self.d
                            )
                        ) for i in range(1, len(self.c)-1)
                    ]
                )
            )
        else:
            self.denses = nn.Sequential(
                OrderedDict(
                    [
                        (
                            f'dense_block_{i}', 
                            self.create_dense(
                                fin=self.c[i],
                                fout=self.c[i+1],
                                d=self.d
                            )
                        ) for i in range(len(self.c)-1)
                    ]
                )
            )

        self.final = nn.Linear(
            self.c[-1], num_classes
        )

    def create_dense(self, fin, fout, d):
        return nn.Sequential(
            nn.Linear(fin, fout),
            nn.Dropout(d),
            nn.LeakyReLU()
        )

    def forward(self, x, pre_head=False, x_2=None, x_sub=None):
        if self.ensemble:
            assert x_2 is not None, "x_2 is None!"
            x = self.pre_dense_1(x)
            x_2 = self.pre_dense_2(x_2)
            x = x + x_2
        
        if self.sub:
            assert x_sub is not None, "x_sub is None!"
            x = self.pre_dense(x)
            x = torch.concat(
                [x, x_sub], dim=1
            )
        
        x = self.denses(x)
        if pre_head:
            return x, self.final(x)
        return self.final(x)
