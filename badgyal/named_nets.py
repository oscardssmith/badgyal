from badgyal.abstractnet import LoadedNet

BGNet = LoadedNet("badgyal-8.pb.gz", 128, 10, 4, cuda=True)
MGNet = LoadedNet("meangirl-8.pb.gz", 32, 4, 2, cuda=True)
