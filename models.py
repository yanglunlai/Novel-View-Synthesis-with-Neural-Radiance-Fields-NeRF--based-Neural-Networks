import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


## TODO : 1. ViewDependent model     (Charles)
##        2. No view dependent model (Alan's)
#
#
#


class NeRF(nn.Module):
    def __init__(self, 
                 N_layers = 4, # number of feature layers in the network
                 channel = 128, # number of channel in a noraml hidden layer
                 pts_channel = 63, # the input positional encoding x vector dimension
                 dir_channel = 27, # the input positional encoding d vector dimension
                 out_channel = 4, # if view_dir = False the output vector is 4x1
                 skip = 2,
                 view_dir = True # whether use viewing direction
                 ):
      super(NeRF, self).__init__()

      self.D = N_layers
      self.W = channel
      self.pts_channel = pts_channel
      self.dir_channel = dir_channel
      self.out_channel = out_channel
      self.skip = skip
      self.use_viewdirs = view_dir

      # Initialize linear layer
      self.feature = [nn.Linear(pts_channel, channel)] # init 1st layer
      for i in range(N_layers-1): # init the remaining layer, N_layers-1 is because the exclusion of the first layer.
        if i != self.skip:
          self.feature += [nn.Linear(self.W, self.W)]
        else:
          self.feature += [nn.Linear(self.W + pts_channel, self.W)]
      self.feat_linear  = nn.ModuleList(self.feature)

      # Initialize directional layer
      self.dir_linear = nn.Linear(self.dir_channel + self.W, self.W//2)
      if self.use_viewdirs: 
        self.feat2_linear = nn.Linear(self.W,self.W)
        self.sigma_linear = nn.Linear(self.W,1)
        self.rgb_linear = nn.Linear(self.W//2,3)
      else:
        self.out_linear = nn.Linear(self.W, out_channel)
        
        
    def forward(self, input):
      # check point:
      if self.D-self.skip < 2:
        print("Error: The skip value must be at least 2 smaller than the number of layers.")

      # load input data
      pts, dirs = input[..., :self.pts_channel], input[..., self.pts_channel:] # split the pts and dirs vector
      x = pts.clone()
      for i in range(self.D):
        out = self.feat_linear[i](x)
        x = torch.relu(out) # add relu after each linear layer
        if i == self.skip: # update before entering the skip layer i=self.skip the one layer before the skip layer
          x = torch.cat([pts, x],-1)
      
      if self.use_viewdirs:
        density  = self.sigma_linear(x)
        density = torch.relu(density)
        x = self.feat2_linear(x) # no relu in this linear layer
        x = torch.cat([x,dirs], -1)
        x = self.dir_linear(x)

        rgb  = self.rgb_linear(x)
        rgb = torch.sigmoid(rgb)

        outputs = torch.cat([rgb, density],-1)
       
      else:
        outputs = self.out_linear(x)

      
      return outputs

class FastNeRF(nn.Module):
    def __init__(self, 
                 N_layers_Fpos = 5,
                 N_layers_Fdir = 3,
                 W_Fpos=256,
                 W_Fdir=128,                 
                 L_pe_pos = 10,
                 L_pe_dir = 4,
                 D = 8,
                 pts_channel = 63, # 63, # the input positional encoding x vector dimension
                 dir_channel = 27, # the input positional encoding d vector dimension
                 out_channel = 4, # if view_dir = False the output vector is 4x1
                 skip = 10, # = N_layers - 4
                 view_dir = True, # whether use viewing direction
                 rgb_use_sigmoid = False,
                 use_concat = False
                 ):
      super(FastNeRF, self).__init__()

      self.N_layers_Fpos = N_layers_Fpos
      self.N_layers_Fdir = N_layers_Fdir
      self.W_Fpos = W_Fpos
      self.W_Fdir = W_Fdir
      self.L_pe_pos = L_pe_pos
      self.L_pe_dir = L_pe_dir
      self.D = D
      self.pts_channel = pts_channel
      self.dir_channel = dir_channel
      self.out_channel = out_channel
      self.skip = skip
      self.use_viewdirs = view_dir
      self.rgb_use_sigmoid = rgb_use_sigmoid
      self.use_concat = use_concat
      # Fpos: 
      # 1 pts_channel => W_Fpos
      # N_layers_Fpos-2 W_Fpos => W_Fpos
      # 1 W_Fpos => 3

      # Input layer
      self.feature_Fpos = [nn.Linear(pts_channel, W_Fpos)]
      for i in range(N_layers_Fpos-2):
        if i == self.skip-1:
            if self.use_concat==True:
                self.feature_Fpos += [nn.Linear(W_Fpos + pts_channel, W_Fpos)]
            else:
                self.feature_Fpos += [nn.Linear(W_Fpos, W_Fpos)]  
        else:
          self.feature_Fpos += [nn.Linear(W_Fpos, W_Fpos)]
      # Output layer
      self.feature_Fpos += [nn.Linear(W_Fpos, 4)]

      # Fdir:
      # 1 dir_channel => W_Fdir
      # N_layers_Fdir-2 WFdir=>WFdir
      # 1 WFdir => D
      self.feature_Fdir = [nn.Linear(dir_channel, W_Fdir)]
      for i in range(N_layers_Fdir-2): # init the remaining layer, N_layers-1 is because the exclusion of the first layer.
        self.feature_Fdir += [nn.Linear(W_Fdir, W_Fdir)]  
      self.feature_Fdir += [nn.Linear(W_Fdir, D)]

      # ModuleLists for Fpos and Fdir  
      self.feature_Fpos  = nn.ModuleList(self.feature_Fpos)   
      self.feature_Fdir = nn.ModuleList(self.feature_Fdir)


    def Fpos(self, x):
        """
        Fpos , Num_layers = 8, W = 384, L_pe =10 (63) (Positional Encoding)
        """
        pts = x.clone()                
        pts = pts.unsqueeze(dim=1).repeat(1, self.D, 1) # (N, D, 63)
        x = pts.clone()                                 # (N, D, 63)

        for i in range(self.N_layers_Fpos):
            out = self.feature_Fpos[i](x)
            x = torch.relu(out) # add relu after each linear layer
            if self.use_concat==True: #should consider skip
                if i == self.skip-1: # update before entering the skip layer i=self.skip the one layer before the skip layer
                    x = torch.cat([pts, x],-1)
        uvw = x
        return uvw

    def Fdir(self, dir):
        """
        Fdir , Num_layers = 4, W = 128, L_pe =4 (27) (Positional Encoding)
        """
        x = dir.clone()
        for i in range(self.N_layers_Fdir):
            out = self.feature_Fdir[i](x)
            x = torch.relu(out) # add relu after each linear layer
        betas = x
        return betas #(N,D)

    def forward(self, input):

        """
        TODO: separate the input point and view
            -args:
            input: (N, pts_channel + dir_channel)

            -return:(output)
                (N, 4)
                (rgb+density)

            # 2. Fdir , Num_layer = 4,  W = 128, L_pe = 4
            # 3. Fpos: pts => (density, (u,v,w) Dx1, D = W = hidden_layer_size )
            # 4. Fdir: dir(theat,phi) => betas (1 x D)
            # 5. D: positional encoding length
            # 6. rgb = beta @ uvw (1xD @ (D,3) => (1,3))    

        """

        pts, dirs = input[..., :self.pts_channel], input[..., self.pts_channel:] # split the pts and dirs vector
        x = pts.clone()

        # Run Fpos network. x: positions                          (N, D, 3)
        duvw = self.Fpos(x)                                     # (N, D, 4)
        density = torch.relu(duvw[...,0])
        uvw = duvw[...,1:]                                      # (N, D, 3)

        # Run Fdir network. dirs:view dirs                        (N, D, 3)
        betas = self.Fdir(dirs)                                  #(N, D)
        betas = betas.reshape(betas.shape[0], betas.shape[1], 1) #(N, D, 1)
        betas = betas.repeat(1,1,3)                              #(N, D, 3)

        # linear combination
        if self.rgb_use_sigmoid:
            rgb = torch.sigmoid(torch.sum(betas * uvw, dim=-2))
        else:
            rgb = (torch.sum(betas * uvw, dim=-2))         # sum over dimension 'D' (N,D,3)

        density = torch.sum(density, dim=-1, keepdim=True) # sum over D (N,D)=>(N,1)
        output = torch.cat((rgb, density), dim=-1)         # (N, 4)
        return output

