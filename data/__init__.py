"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    # importlib.import_module这个函数的作用类似于Python中的import语句，但是它可以在运行时根据字符串动态导入模块，而不是在代码编写阶段就确定导入的模块
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    """
    在这段代码中，datasetlib.dict.items()是一个遍历数据集模块字典的循环。让我来详细解释一下：
    datasetlib是一个模块对象，通过importlib.import_module动态导入得到。这个模块对象包含了被导入模块的所有内容，比如类、函数、变量等。
    datasetlib.__dict__是模块对象的一个属性，它是一个字典，包含了模块中定义的所有内容。字典的键是内容的名称，值是内容的对象。
    datasetlib.__dict__.items()是字典的items方法，用于返回字典中的键值对。在这段代码中，它被用于遍历模块字典中的所有内容。
    for name, cls in datasetlib.__dict__.items():是一个循环语句，它遍历了datasetlib.__dict__.items()返回的键值对。
    在每次循环中，name是字典中的键，表示模块中定义的内容的名称；cls是字典中的值，表示对应的内容的对象，可以是类、函数、变量等。
    """
    for name, cls in datasetlib.__dict__.items():
        """
        name.lower() == target_dataset_name.lower()：这个条件检查当前项的名称（经过小写转换后）是否与目标数据集名称（经过小写转换后）相等。这样做是为了不区分大小写地进行匹配。
        issubclass(cls, BaseDataset)：这个条件检查当前项的值cls是否是BaseDataset的子类。issubclass是一个Python内置函数，用于检查一个类是否是另一个类的子类。
        只有当这两个条件都满足时，即当前项的名称与目标数据集名称匹配且当前项的值是BaseDataset的子类时，dataset变量才会被赋值为当前项的值cls。
        """
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_name)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        drop_last = True if opt.isTrain else False
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads), drop_last=drop_last)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
