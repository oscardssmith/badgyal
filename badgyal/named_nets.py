from badgyal.abstractnet import LoadedNet

BGNet = lambda cuda: LoadedNet("badgyal-8.pb.gz", 128, 10, 4, cuda=cuda)
MGNet = lambda cuda: LoadedNet("meangirl-8.pb.gz", 32, 4, 2, cuda=cuda)
