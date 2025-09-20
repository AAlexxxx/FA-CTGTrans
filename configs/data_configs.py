def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class CTU_CHB():
    def __init__(self):
        super(CTU_CHB, self).__init__()
        # data parameters
        self.num_classes = 2
        self.class_names = ['0', '1']
        self.sequence_len = 7200

        # model configs
        self.input_channels = 2
        self.kernel_size = 8
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 32
        self.final_out_channels = 128

        # Transformer
        self.trans_dim = 902
        self.num_heads = 2


