from math import inf
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
import pandas as pd
from tqdm import tqdm
from data_utils.person_activity import PersonActivity
from data_utils.physionet import PhysioNet

Constants_PAD = 0

def get_data_min_max(records, device):
	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals).to(device))
				batch_max.append(torch.max(non_missing_vals).to(device))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max

def proc_hii_data(x, y, input_dim, args):
    x = x[:, :input_dim*2+1]

    if args.debug_flag:
        x = x[:1000, :]
        y = y[:1000]
        
    if args.task == "los":
        y = y-1

    x = np.transpose(x, (0, 2, 1))
    
    new_x = np.empty((len(x), len(x[1]), input_dim*3+1))
        
        
    print("data preprocessing in batch...")
    total = len(x)
    batch_sz = 20000

    pbar = tqdm(range(0, total, batch_sz)) 
    for start in pbar:
        end = min(start+batch_sz, total)
        
        new_x[start:end, :, :input_dim*2+1] = process_data(x[start:end], input_dim)
        new_x[start:end, :, input_dim*2+1:input_dim*3+1] = cal_tau(x[start:end, :, -1], x[start:end, :, input_dim:2 * input_dim])


    print("data preprocess in batch done.")
    return new_x, y

def proc_hii_set_data(x, y, input_dim, args):
    x = x[:, :input_dim*2+1]

    if args.debug_flag:
        x = x[:1000, :]
        y = y[:1000]
        
    if args.task == "los":
        y = y-1

    x = np.transpose(x, (0, 2, 1))
    
    if args.enc_tau is not None:
        new_x = np.empty((len(x), len(x[1]), input_dim*3+1))
    else:
        new_x = np.empty((len(x), len(x[1]), input_dim*2+1))
        
        
    print("data preprocessing in batch...")
    total = len(x)
    batch_sz = 100

    pbar = tqdm(range(0, total, batch_sz)) 
    for start in pbar:
        end = min(start+batch_sz, total)

        new_x[start:end, :, :input_dim*2+1] = process_data(x[start:end], input_dim)

        if args.enc_tau is not None:
            new_x[start:end, :, input_dim*2+1:input_dim*3+1] = cal_tau(x[start:end, :, -1], x[start:end, :, input_dim:2 * input_dim])

    print("data preprocess in batch done.")
    return new_x, y

def get_clints_hii_data(args, to_set=False):
    
    if args.task == "vent" or args.task == "vaso":
        data_folder_x = args.root_path + args.data_path + 'cip/' 
        data_folder_y = args.root_path + args.data_path + 'cip/' + args.task + '_'
    elif args.task == "pretrain":
        data_folder_x = args.root_path + args.data_path + 'mor/' 
        data_folder_y = args.root_path + args.data_path + 'mor/' 
    else:
        data_folder_x = args.root_path + args.data_path + args.task + '/' 
        data_folder_y = args.root_path + args.data_path + args.task + '/' 
    
    dataloader = []

    for set_name in ['train', 'val', 'test']:
        data_x_all = []
        data_y_all = []
        
        if set_name == 'train':
            shuffle = True
        else:
            shuffle = False
            
        print("loading " + set_name + " data")
        if set_name == "train" and args.task != "mor" and args.load_in_batch:
            for i in range(5):
                data_x = np.load(data_folder_x + set_name + '_input' + str(i) + '.npy', allow_pickle=True)
                data_y = np.load(data_folder_y + set_name + '_output' + str(i) + '.npy', allow_pickle=True)
                
                args.num_types = int((data_x.shape[1] - 1) / 2)
                data_x, data_y = proc_hii_data(data_x, data_y, args.num_types, args)
                data_x_all.append(data_x)
                data_y_all.append(data_y)
                del data_x, data_y
            
            data_x_all = np.concatenate(data_x_all)
            data_y_all = np.concatenate(data_y_all)
            
        else:
            data_y_all = np.load(data_folder_y + set_name + '_output.npy', allow_pickle=True)
            data_x_all = np.load(data_folder_x + set_name + '_input.npy', allow_pickle=True)
                
            args.num_types = int((data_x_all.shape[1] - 1) / 2)
            data_x_all, data_y_all = proc_hii_data(data_x_all, data_y_all, args.num_types, args)

        print(data_x_all.shape, data_y_all.shape)
        dataloader.append(get_data_loader(data_x_all, data_y_all, args, shuffle=shuffle))
        del data_x_all, data_y_all
        
    print("type num: ", args.num_types)
    return dataloader[0], dataloader[1], dataloader[2]


def get_data_loader(data_x, data_y, args, shuffle=False):
    data_combined = TensorDataset(torch.from_numpy(data_x).float(),
                                        torch.from_numpy(data_y).long().squeeze())
    dataloader = DataLoader(
        data_combined, batch_size=args.batch_size, shuffle=shuffle,num_workers=8)
    
    return dataloader


def cal_tau(observed_tp, observed_mask):
    # input [B,L,K], [B,L]
    # return [B,L,K]
    # observed_mask, observed_tp = x[:, :, input_dim:2 * input_dim], x[:, :, -1]
    if observed_tp.ndim == 2:
        tmp_time = observed_mask * np.expand_dims(observed_tp,axis=-1) # [B,L,K]
    else:
        tmp_time = observed_tp.copy()
        
    b,l,k = tmp_time.shape
    
    new_mask = observed_mask.copy()
    new_mask[:,0,:] = 1
    tmp_time[new_mask == 0] = np.nan
    tmp_time = tmp_time.transpose((1,0,2)) # [L,B,K]
    tmp_time = np.reshape(tmp_time, (l,b*k)) # [L, B*K]

    # padding the missing value with the next value
    df1 = pd.DataFrame(tmp_time)
    df1 = df1.fillna(method='ffill')
    tmp_time = np.array(df1)

    tmp_time = np.reshape(tmp_time, (l,b,k))
    tmp_time = tmp_time.transpose((1,0,2)) # [B,L,K]
    
    tmp_time[:,1:] -= tmp_time[:,:-1]
    del new_mask
    return tmp_time * observed_mask


def process_data(x, input_dim, m=None, tt=None, x_only=False):
    if not x_only:
        observed_vals, observed_mask, observed_tp = x[:, :,
                                                    :input_dim], x[:, :, input_dim:2 * input_dim], x[:, :, -1]
        observed_tp = np.expand_dims(observed_tp, axis=-1)
    else:
        observed_vals = x
        assert m is not None
        observed_mask = m
        observed_tp = tt

    observed_vals = tensorize_normalize(observed_vals)
    observed_vals[observed_mask == 0] = 0
    if not x_only:
        return np.concatenate((observed_vals, observed_mask, observed_tp), axis=-1)
    return observed_vals


def tensorize_normalize(P_tensor):
    mf, stdf = getStats(P_tensor)
    P_tensor = normalize(P_tensor, mf, stdf)
    return P_tensor

def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        if len(vals_f) > 0:
            mf[f] = np.mean(vals_f)
            tmp_std = np.std(vals_f)
            stdf[f] = np.max([tmp_std, eps])
    return mf, stdf

def normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    return Pnorm_tensor

def variable_time_collate_fn(batch, device, input_dim, return_np=False,to_set=False,maxlen=None,
                             data_min=None, data_max=None, activity=False):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    # number of labels
    # N = batch[0][-1].shape[1] if activity else 1
    if maxlen is None:
        len_tt = [ex[1].size(0) for ex in batch]
        maxlen = np.max(len_tt)

    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)

    if activity:
        combined_labels = torch.zeros([len(batch), maxlen]).to(device)
    else:
        combined_labels = torch.zeros([len(batch)]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = min(tt.size(0),maxlen)
        enc_combined_tt[b, :currlen] = tt[:currlen].to(device)
        enc_combined_vals[b, :currlen] = vals[:currlen].to(device)
        enc_combined_mask[b, :currlen] = mask[:currlen].to(device)

        if labels.dim() == 2:
            combined_labels[b] = torch.argmax(labels,dim=-1)
        else:
            combined_labels[b] = labels.to(device)

    enc_combined_vals = torch.tensor(process_data(
                                        enc_combined_vals.cpu().numpy(), 
                                        m=enc_combined_mask.cpu().numpy(), 
                                        tt=enc_combined_tt,
                                        input_dim=input_dim, 
                                        x_only=True)).to(enc_combined_tt.device)


    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)


    tau = torch.tensor(cal_tau(enc_combined_tt.cpu().numpy(), enc_combined_mask.cpu().numpy())).to(enc_combined_vals.device)
    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1), tau), 2)


    return combined_data, combined_labels


def get_activity_data(args, device):
    n_samples = 8000
    dataset_obj = PersonActivity(args.data_path + 'PersonActivity',
                                 download=True, n_samples=n_samples, device=device)

    print(dataset_obj)

    train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8,
                                                             random_state=42, shuffle=False)

    record_id, tt, vals, mask, labels = train_data[0]
    input_dim = vals.size(-1)
    args.num_types = input_dim

    batch_size = min(len(dataset_obj), args.batch_size)
    
    if not args.retrain:
        train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                random_state=11, shuffle=False)

        val_data_combined = variable_time_collate_fn(val_data, device, input_dim=input_dim,activity=True)
        val_data_combined = TensorDataset(
            val_data_combined[0], val_data_combined[1].long())

        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None
    
    train_data_combined = variable_time_collate_fn(train_data, device, input_dim=input_dim,activity=True)
    test_data_combined = variable_time_collate_fn(test_data, device, input_dim=input_dim,activity=True)

    # norm_mean = train_data_combined[0][:, :, :input_dim].mean(dim=0, keepdim=True).cpu()
    
    print(train_data_combined[1].sum(), test_data_combined[1].sum())

    print(train_data_combined[0].size(), train_data_combined[1].size(),
          test_data_combined[0].size(), test_data_combined[1].size())

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long())
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long())

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)


    return train_dataloader, val_dataloader, test_dataloader, input_dim



def get_physionet_data(args, device, q=0.016, flag=1):
    train_dataset_obj = PhysioNet(args.data_path + 'physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=8000,
                                  device=device)

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]
    data_min, data_max = get_data_min_max(total_dataset, device)
    print(len(total_dataset))
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]


    input_dim = vals.size(-1)
    batch_size = min(len(train_dataset_obj), args.batch_size)
    args.num_types = input_dim

    if not args.retrain:
        train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                random_state=11, shuffle=True)

        val_data_combined = variable_time_collate_fn(val_data, device,input_dim=input_dim,data_min=data_min, data_max=data_max)

        val_data_combined = TensorDataset(
            val_data_combined[0], val_data_combined[1].long().squeeze())

        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=True)
    else:
        val_dataloader = None

    train_data_combined = variable_time_collate_fn(train_data, device,input_dim=input_dim,data_min=data_min, data_max=data_max)
    test_data_combined = variable_time_collate_fn(test_data, device,input_dim=input_dim,data_min=data_min, data_max=data_max)

    # norm_mean = train_data_combined[0][:, :, :input_dim].mean(dim=0, keepdim=True).cpu()

    print(train_data_combined[1].sum(
    ), test_data_combined[1].sum())
    print(train_data_combined[0].size(), train_data_combined[1].size(),
            test_data_combined[0].size(), test_data_combined[1].size())

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long().squeeze())
    
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long().squeeze())

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)
    

    return train_dataloader, val_dataloader, test_dataloader, input_dim


def cal_label_freq(labels):
    freq_dict = {}
    for i in labels:
        if i not in freq_dict:
            freq_dict[i] = sum(labels==i)

    print(freq_dict)


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max
