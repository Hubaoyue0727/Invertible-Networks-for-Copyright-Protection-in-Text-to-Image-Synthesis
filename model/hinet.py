from model import *
from model.invblock import INV_block
import torch.nn as nn
class Hinet_stage(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self):
        super(Hinet_stage, self).__init__()

        self.inv1 = INV_block()
        self.inv2 = INV_block()
        self.inv3 = INV_block()
        self.inv4 = INV_block()
        self.inv5 = INV_block()
        self.inv6 = INV_block()
        self.inv7 = INV_block()
        self.inv8 = INV_block()

        self.inv9 = INV_block()
        self.inv10 = INV_block()
        self.inv11 = INV_block()
        self.inv12 = INV_block()
        self.inv13 = INV_block()
        self.inv14 = INV_block()
        self.inv15 = INV_block()
        self.inv16 = INV_block()

        self.inv17 = INV_block()
        self.inv18 = INV_block()
        self.inv19 = INV_block()
        self.inv20 = INV_block()
        self.inv21 = INV_block()
        self.inv22 = INV_block()
        self.inv23 = INV_block()
        self.inv24 = INV_block()

        # self.inv25 = INV_block()
        # self.inv26 = INV_block()
        # self.inv27 = INV_block()
        # self.inv28 = INV_block()
        # self.inv29 = INV_block()
        # self.inv30 = INV_block()
        # self.inv31 = INV_block()
        # self.inv32 = INV_block()

    def forward(self, x, rev=False, advinn=False):
        # if advinn:
        #     out = self.inv1(x)
        #     out = self.inv2(out)
            
        # else:
        if not rev:
            
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)

            out = self.inv17(out)
            out = self.inv18(out)
            out = self.inv19(out)
            out = self.inv20(out)
            out = self.inv21(out)
            out = self.inv22(out)
            out = self.inv23(out)
            out = self.inv24(out)  

            # out = self.inv25(out)
            # out = self.inv26(out)
            # out = self.inv27(out)
            # out = self.inv28(out)
            # out = self.inv29(out)
            # out = self.inv30(out)
            # out = self.inv31(out)
            # out = self.inv32(out)

        else:
            # out = self.inv32(x, rev=True)
            # out = self.inv31(out, rev=True)
            # out = self.inv30(out, rev=True)
            # out = self.inv29(out, rev=True)
            # out = self.inv28(out, rev=True)
            # out = self.inv27(out, rev=True)
            # out = self.inv26(out, rev=True)
            # out = self.inv25(out, rev=True)
        
            out = self.inv24(x, rev=True)
            out = self.inv23(out, rev=True)
            out = self.inv22(out, rev=True)
            out = self.inv21(out, rev=True)
            out = self.inv20(out, rev=True)
            out = self.inv19(out, rev=True)
            out = self.inv18(out, rev=True)
            out = self.inv17(out, rev=True)
        
            print('-------------------------------------- now rev !! -------------------------------------')
            out = self.inv16(out, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out
