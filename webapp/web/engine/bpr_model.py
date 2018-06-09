import json

class BPRModel(object):
    def __init__(self):
        super(BPRModel, self).__init__()
        with open('./data/reclist_json') as json_data:
            self.bprmodel = json.loads(json.load(json_data))

    def get_bpr_matrix(self):
        return self.bprmodel