
# from torch import nn

#----------------------------------------------------------------------
from dataset import Dataset
from ..utils import Config, LogClient, LogServer, mem_report
from ..models import GraphConvolutionalModel
#----------------------------------------------------------------------


import torch.distributed  as dist
from torch.nn.parallel import DistributedDataParallel
import argparse
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
# from algorithms.models import GraphConvolutionalModel
from torch.optim import Adam
import numpy as np

# def get_args():
#     parser = argparse.ArgumentParser(description="Train program use DDP")
#
#     # ddp运行所需参数，请勿随意修改
#     parser.add_argument("--local_rank", type=int ,default=-1, help="local_rank for ddp")
#     # ddp训练过程中意外终止代码可能导致进程驻留，about参数帮助查看相关进程，并杀死
#     parser.add_argument("--about", type=str, default="ddp", help="description for ddp process")
#     # 观察程序运行过程中显存的使用情况，显存空余较大情况，请提高batch_size
#     parser.add_argument("--batch_size", type=int, default=16, help="batch size")
#     # 数据集加载过程中多线程参数，参数越大，数据加载越快，可根据cpu核心数调整
#     parser.add_argument("--workers", type=int, default=8, help="num woekers")
#     parser.add_argument("--save_path", type=str, default="./checkpoints", help="model save path")
#     parser.add_argument("--epochs", type=int, default=1000, help="mutil train model")
#
#     args = parser.parser_args()
#     return args


#这部分是模型，主要看train里面返回的部分就行了，主要是计算了loss，然后返回loss

class Model(nn.Module):

    def __init__(self,logger,adj,observation_dim,action_dim,n_agent,p_args,data,args):
        # ddp部分
        self.args = args
        self.device = torch.device("cuda", args.local_rank)
        self.model = GraphConvolutionalModel(logger, adj, observation_dim, action_dim, n_agent, p_args).to(self.device)
        self.model.to(self.device)
        self.optimizer_p = Adam(self.model.parameters(), lr=self.lr)

        # 此处的model是你数据集加载部分产生数据的model，请勿混淆
        # 下列代码为数据集加载代码，所需参数根据自身需要修改
        # self.train_data = Dataset(model, samples_num=2000, batch_size=256)
        self.train_data = data
        sample = DistributedSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data, num_workers=args.workers, batch_size=args.batch_size, sampler=sample, shuffle=False)


    def updateModel(self, length=1):
        min_train_loss = 10000
        
        for epoch in range(self.args.epochs):
        #ss,actions,rs, sls, ds就是数据，是tensor/list的形式，ss, actions, rs, s1s, ds = [...], [...], [...], [...],[...]
        #这里是train模型
            if self.args.local_rank != -1:
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True)

                train_loss = 0
                self.model.train()

            for step, train_batch in enumerate(self.train_dataloader):
                ss, actions, rs, s1s, ds = train_batch

                loss, rel_state_error = self.model.train(ss, actions, rs, s1s, ds, length) # [n_traj, T, n_agent, dim]
            
                self.optimizer_p.zero_grad()
                loss.sum().backward()
                self.optimizer_p.step()
                train_loss += loss.item() * len(train_batch)

            train_loss /= len(self.train_dataloader)
            print("Epoch: %d, train_loss: %.6f" % (epoch, train_loss))
            if self.args.local_rank == 0:
                if train_loss < min_train_loss:
                    save_file = '%s/DG_rotate_epoch_%04d_loss_%.6f.pth' % (self.args.save_path, epoch, train_loss)
                    torch.save(self.model.state_dict(), save_file)
                    min_train_loss = train_loss


        # return rel_state_error.item()


if __name__ == "__main__":
    


    
    
    def get_args():
        parser = argparse.ArgumentParser(description="Train program use DDP")

        # ddp运行所需参数，请勿随意修改
        parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for ddp")
        # ddp训练过程中意外终止代码可能导致进程驻留，about参数帮助查看相关进程，并杀死
        parser.add_argument("--about", type=str, default="ddp", help="description for ddp process")
        # 观察程序运行过程中显存的使用情况，显存空余较大情况，请提高batch_size
        parser.add_argument("--batch_size", type=int, default=16, help="batch size")
        # 数据集加载过程中多线程参数，参数越大，数据加载越快，可根据cpu核心数调整
        parser.add_argument("--workers", type=int, default=8, help="num woekers")
        parser.add_argument("--save_path", type=str, default="./checkpoints", help="model save path")
        parser.add_argument("--epochs", type=int, default=2000, help="mutil train model")

        args = parser.parser_args()
        return args


    args = get_args()
    args.device = "cuda:%s" % args.local_rank
    
    torch.cuda.set_device(args.device)
    
    #需要传入的数据 length,logger,adj,observation_dim,action_dim,n_agent,p_args,data
    #-----------------------------------------------------------------------   
    data = Dataset(1, samples_num=2000, batch_size=256)
    adj = torch.ones(28,28)
    observation_dim = 10
    action_dim = 1
    n_agent = 28

    p_args = Config()
    p_args.n_conv = 1
    p_args.n_embedding = 0
    p_args.residual = True
    p_args.edge_embed_dim = 12
    p_args.node_embed_dim = 8
    p_args.edge_hidden_size = [16, 16]
    p_args.node_hidden_size = [16, 16]
    p_args.reward_coeff = 1.
    agent_args.p_args = p_args
    p_arg = agent_args.p_args  
    logger = LogServer({'run_args':123, 'algo_args':123}, mute=True)
    logger = LogClient(logger)
    #----------------------------------------------------------
    
    

    dist.init_process_group(backend="nccl", init_method="env://")

    model = Model(logger,adj,observation_dim,action_dim,n_agent,p_arg,data,args)

    model.update(length)

    if args.local_rank == 0:
        dist.destroy_process_group()

    torch.cuda.empty_cache()

    