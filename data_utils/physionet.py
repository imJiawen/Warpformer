import os
# import utils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url


def get_data_min_max(records, device):
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max


class PhysioNet(object):

  urls = [
    'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
    'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
  ]

  outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

  params = [
    'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
    'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
  ]

  params_dict = {k: i for i, k in enumerate(params)}

  labels = [ "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death" ]
  labels_dict = {k: i for i, k in enumerate(labels)}

  def __init__(self, root, train=True, download=False,
    quantization = 0.1, n_samples = None, device = torch.device("cpu")):

    self.root = root
    self.train = train
    self.device = device
    self.reduce = "average"
    self.quantization = quantization

    if download:
      self.download()

    if not self._check_exists():
      raise RuntimeError('Dataset not found. You can use download=True to download it')

    if self.train:
      data_file = self.training_file
    else:
      data_file = self.test_file
    
    if self.device == 'cpu':
      self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location='cpu')
      self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), map_location='cpu')
    else:
      self.data = torch.load(os.path.join(self.processed_folder, data_file))
      self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

    if n_samples is not None:
      self.data = self.data[:n_samples]
      self.labels = self.labels[:n_samples]


  def download(self):
    if self._check_exists():
      return

    #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(self.raw_folder, exist_ok=True)
    os.makedirs(self.processed_folder, exist_ok=True)

    # Download outcome data
    for url in self.outcome_urls:
      filename = url.rpartition('/')[2]
      download_url(url, self.raw_folder, filename, None)

      txtfile = os.path.join(self.raw_folder, filename)
      with open(txtfile) as f:
        lines = f.readlines()
        outcomes = {}
        for l in lines[1:]:
          l = l.rstrip().split(',')
          record_id, labels = l[0], np.array(l[1:]).astype(float)
          outcomes[record_id] = torch.Tensor(labels).to(self.device)

        torch.save(
          labels,
          os.path.join(self.processed_folder, filename.split('.')[0] + '.pt')
        )

    for url in self.urls:
      filename = url.rpartition('/')[2]
      download_url(url, self.raw_folder, filename, None)
      tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
      tar.extractall(self.raw_folder)
      tar.close()

      print('Processing {}...'.format(filename))

      dirname = os.path.join(self.raw_folder, filename.split('.')[0])
      patients = []
      total = 0
      for txtfile in os.listdir(dirname):
        record_id = txtfile.split('.')[0]
        with open(os.path.join(dirname, txtfile)) as f:
          lines = f.readlines()
          prev_time = 0
          tt = [0.]
          vals = [torch.zeros(len(self.params)).to(self.device)]
          mask = [torch.zeros(len(self.params)).to(self.device)]
          nobs = [torch.zeros(len(self.params))]
          for l in lines[1:]:
            total += 1
            time, param, val = l.split(',')
            # Time in hours
            time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
            # round up the time stamps (up to 6 min by default)
            # used for speed -- we actually don't need to quantize it in Latent ODE
            time = round(time / self.quantization) * self.quantization

            if time != prev_time:
              tt.append(time)
              vals.append(torch.zeros(len(self.params)).to(self.device))
              mask.append(torch.zeros(len(self.params)).to(self.device))
              nobs.append(torch.zeros(len(self.params)).to(self.device))
              prev_time = time

            if param in self.params_dict:
              #vals[-1][self.params_dict[param]] = float(val)
              n_observations = nobs[-1][self.params_dict[param]]
              if self.reduce == 'average' and n_observations > 0:
                prev_val = vals[-1][self.params_dict[param]]
                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                vals[-1][self.params_dict[param]] = new_val
              else:
                vals[-1][self.params_dict[param]] = float(val)
              mask[-1][self.params_dict[param]] = 1
              nobs[-1][self.params_dict[param]] += 1
            else:
              assert param == 'RecordID', 'Read unexpected param {}'.format(param)
        tt = torch.tensor(tt).to(self.device)
        vals = torch.stack(vals)
        mask = torch.stack(mask)

        labels = None
        if record_id in outcomes:
          # Only training set has labels
          labels = outcomes[record_id]
          # Out of 5 label types provided for Physionet, take only the last one -- mortality
          labels = labels[4]

        patients.append((record_id, tt, vals, mask, labels))

      torch.save(
        patients,
        os.path.join(self.processed_folder, 
          filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
      )
        
    print('Done!')

  def _check_exists(self):
    for url in self.urls:
      filename = url.rpartition('/')[2]

      if not os.path.exists(
        os.path.join(self.processed_folder, 
          filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
      ):
        return False
    return True

  @property
  def raw_folder(self):
    return os.path.join(self.root, self.__class__.__name__, 'raw')

  @property
  def processed_folder(self):
    return os.path.join(self.root, self.__class__.__name__, 'processed')

  @property
  def training_file(self):
    return 'set-a_{}.pt'.format(self.quantization)

  @property
  def test_file(self):
    return 'set-b_{}.pt'.format(self.quantization)

  @property
  def label_file(self):
    return 'Outcomes-a.pt'

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)

  def get_label(self, record_id):
    return self.labels[record_id]

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
    fmt_str += '    Root Location: {}\n'.format(self.root)
    fmt_str += '    Quantization: {}\n'.format(self.quantization)
    fmt_str += '    Reduce: {}\n'.format(self.reduce)
    return fmt_str
