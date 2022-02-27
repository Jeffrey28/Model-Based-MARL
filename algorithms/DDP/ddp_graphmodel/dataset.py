import torch
from tqdm import tqdm


# class Dataset:
#     def __init__(self, model, samples_num, batch_size):
#         self.model = model
#         self.samples_num = samples_num
#         self.batch_size = batch_size
#         self.data = []
#         self.data = self.dataGenerate()

#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return self.samples_num

#     def dataGenerate(self):
#         ss, actions, rs, s1s, ds = [], [], [], [], []
#         for i in tqdm(range(self.samples_num)):
#             trajs = self.model.sampleTrajs(self.batch_size)
#             trajs = [traj.getFraction(length=self.model_update_length) for traj in trajs]

#             for traj in trajs:
#                 s, a, r, s1, d = trajs["s"], traj["a"], trajs["r"], traj["s1"], traj["d"]
#                 s, a, r, s1, d = [torch.as_tensor(item) for item in [s, a, r, s1, d]]
#                 ss.append(s)
#                 actions.append(a)
#                 rs.append(r)
#                 s1s.append(s1)
#                 ds.append(d)
#             self.data.append([ss, actions, rs, s1s, ds])
            

class Dataset:
    #def __init__(self, model_buffer, samples_num, batch_size, length):
    def __init__(self):
        self.model_buffer = model_buffer
        self.samples_num = samples_num
        self.batch_size = batch_size
        self.model_update_length = length
        self.data = 
        self.dataGenerate()
        # self.data = self.dataGenerate()
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.samples_num

    def dataGenerate(self):
        
        ss, actions, rs, s1s, ds = [], [], [], [], []
        
        ss = torch.ones(256,4, 28,10)
        actions = torch.ones(256,4, 28,1)
        rs = torch.ones(256,4, 28,1)
        s1s = torch.ones(256,4, 28,10)
        ds = torch.ones(256,4, 28,1)
        for i in range(200):
            self.data.append([ss, actions, rs, s1s, ds])            


    # def dataGenerate(self):
    #     ss, actions, rs, s1s, ds = [], [], [], [], []
    #     for i in range(200):


    #         for i in range(256):
    #             # s, a, r, s1, d = trajs["s"], traj["a"], trajs["r"], traj["s1"], traj["d"]
    #             # s, a, r, s1, d = [torch.as_tensor(item) for item in [s, a, r, s1, d]]

    #             s = torch.ones(4,28,10)
    #             a = torch.ones(4,28,1)
    #             r = torch.ones(4,28,1)
    #             s1 = torch.ones(4,28,10)
    #             d = torch.ones(4,28,1)


    #             ss.append(s)
    #             actions.append(a)
    #             rs.append(r)
    #             s1s.append(s1)
    #             ds.append(d)
    #         self.data.append([ss, actions, rs, s1s, ds])





