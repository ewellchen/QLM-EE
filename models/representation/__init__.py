
from models.representation.EntangledNN import EntangledNN

from models.representation.LocalMixtureNN import LocalMixtureNN

from models.representation.EntangledNNreal import EntangledNNreal


def setup(opt):
    print("Network type: " + opt.network_type + ' ' + 'net')
    if opt.network_type == "Entangled":
        model = EntangledNN(opt)
    elif opt.network_type == "Mixture":
        model = LocalMixtureNN(opt)
    elif opt.network_type == "Entangled_real":
        model = EntangledNNreal(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
