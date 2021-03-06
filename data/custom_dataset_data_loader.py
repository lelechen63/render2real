import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.datasetname == 'fs':
        from data.facescape_dataset import FacescapeDataset
        dataset = FacescapeDataset()
    elif opt.datasetname == 'fs_pair':
        from data.facescape_paired_dataset import FacescapeDirDataset
        dataset = FacescapeDirDataset()
    elif opt.datasetname == 'fs_texmesh':
        from data.facescape_paired_dataset import FacescapeMeshTexDataset
        dataset = FacescapeMeshTexDataset()
    elif opt.datasetname == 'fs_tex':
        from data.facescape_paired_dataset import FacescapeTexDataset
        dataset = FacescapeTexDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
