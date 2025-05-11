import numpy as np
import torch
import torch.nn as nn

def chiAB(X):
    a = torch.tensor([0.5526,-0.0935])
    b = torch.tensor([0.7184,1.1607])
    rA = torch.tensor(0.1034)
    rB = torch.tensor(0.07275)
    m = nn.Tanh()
    sizex, nothing = X.shape
    chiA = 0.5 - 0.5*m(1000*((((X - a).pow(2)).sum(dim = 1).reshape(sizex,1))-(rA + torch.tensor(0.02)).pow(2)))                     
    chiB = 0.5 - 0.5*m(1000*((((X - b).pow(2)).sum(dim = 1).reshape(sizex,1))-(rB + torch.tensor(0.02)).pow(2)))       
                             
    return chiA, chiB

def q_theta(X,chiA,chiB,q_tilde, device):
    Q = (torch.tensor([1]).to(device) - chiA)*(q_tilde*(torch.tensor([1]).to(device) - chiB)+chiB)
    return Q

class LJ7_2(nn.Module):
    """Feedfoward neural network with 2 hidden layer"""
    def __init__(self, in_size, hidden_size,hidden_size2, out_size):
        super().__init__()
        # 1st hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # 2nd hidden layer
        self.linear2 = nn.Linear(hidden_size,hidden_size2)
        # output layer
        self.linear3 = nn.Linear(hidden_size2, out_size)
        
    def forward(self, xb):
        # Get information from the data
#         xb = torch.cat((torch.sin(xb),torch.cos(xb)),dim = 1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        tanhf = nn.Tanh()
        out = tanhf(out)
        # Get predictions using output layer
        out = self.linear2(out)
        # apply activation function again
        out = tanhf(out)
        # last hidden layer 
        out = self.linear3(out)
        #sigmoid function
        out = torch.sigmoid(out)
        return out
    
    
def chiAB_1D(trail, X, device):
    # A: trapezoid
    # B: hexagon
    """
    trail 0 parameters: 
    b = torch.tensor([0.7085])
    a = torch.tensor([1.8071])
    rB = torch.tensor(0.1979)
    rA = torch.tensor(0.2429)
    
    trail 1 parameters:
    b = torch.tensor([0.2832])    
    a = torch.tensor([2.2])
    rB = torch.tensor(0.45) 
    rA = torch.tensor(0.2)
    
    trail 2 parameters:
    b = torch.tensor([0.2832])    
    a = torch.tensor([2.5])
    rB = torch.tensor(0.15) 
    rA = torch.tensor(0.1)
    
    """ 
    # if trail == 0:
    #     b = torch.tensor([0.7085])
    #     a = torch.tensor([1.8071])
    #     rB = torch.tensor(0.1979)
    #     rA = torch.tensor(0.2429)
    # elif trail == 1:
    #     b = torch.tensor([0.2832])    
    #     a = torch.tensor([2.5])
    #     rB = torch.tensor(0.35) 
    #     rA = torch.tensor(0.1)
    # elif trail == 2:
    #     b = torch.tensor([0.2832])
    #     a = torch.tensor([2.5])
    #     rB = torch.tensor(0.15)
    #     rA = torch.tensor(0.1)

    b = torch.tensor([0.0])    
    a = torch.tensor([3.0])
    rB = torch.tensor(0.8) 
    rA = torch.tensor(1.0)
    
    b = b.to(device)
    a = a.to(device)
    rB = rB.to(device)
    rA = rA.to(device)
    
    m = nn.Tanh()
    if torch.Tensor.dim(X) == 1:
        sizex = 1
        chiA = 0.5 - 0.5*m(1000*(((X - a).pow(2))-(rA + torch.tensor(0.02)).pow(2)))                     
        chiB = 0.5 - 0.5*m(1000*((X - b)-(rB + torch.tensor(0.02))))
    else:
        sizex, nothing = X.shape
        chiA = 0.5 - 0.5*m(1000*((((X - a).pow(2)).sum(dim = 1).reshape(sizex,1))-(rA + torch.tensor(0.02)).pow(2)))                     
        chiB = 0.5 - 0.5*m(1000*(((X - b).reshape(sizex,1))-(rB + torch.tensor(0.02))))
    
    return chiA, chiB

def biased_MALAstep_controlled(x,pot_x,grad_x,q,grad_q,fpot,fgrad,beta,dt, device):
    std = torch.sqrt(2*dt/beta)    
    w = np.random.normal(0.0,std,np.shape(x))
    """
    When the point is too close to A, derivQ = 0, Q = 0 return None
    """
    control = torch.tensor(2)/beta*(grad_q/q)

    # print('x shape: {}, grad_x shape: {}, control shape: {}'.format(x.shape, grad_x.shape, control.shape))
    y = x - (grad_x - control)*dt + torch.tensor(w).to(device)
    pot_y = fpot(y)
    grad_y = fgrad(y, device)
    qxy =  torch.sum(torch.tensor(w)**2)  #||w||^2
    qyx = torch.sum((x - y + dt*grad_y)**2) # ||x-y+dt*grad V(y)||
    alpha = torch.exp(-beta*(pot_y-pot_x+(qyx-qxy)*0.25/dt))
#     x = y
#     pot_x = pot_y
#     grad_x = grad_y
    if alpha >= 1: # accept move 
        x = y
        pot_x = pot_y
        grad_x = grad_y
    else:    
        eta = np.random.uniform(0.0,1.0,(1,))
        if eta < alpha.cpu().detach().numpy(): # accept move 
            x = y
            pot_x = pot_y
            grad_x = grad_y
            # print("ACCEPT: alpha = ",alpha," eta = ",eta)
        else:
            pass
    return x,pot_x,grad_x   
    

