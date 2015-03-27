from predictor import GenderPredictor

class DataAnalysis(GenderPredictor):

    def __init__(self):
        super(DataAnalysis, self).__init__()

    def _loadTrainingData(self, selected_id=0):
        count_id = [{} for i in xrange(4)]
        count_id_gender = [{} for i in xrange(4)]
        count_len = {}
        genders = self._loadTrainingLabels()
        with open('data/trainingData.csv') as f:
            line_count = -1
            for line in f.readlines():
                line_count += 1
                d = line.strip().split(',')
                products = d[3]
                label = genders[line_count]
                ids = products.split(';')
                count_len[len(ids)] = count_len.get(len(ids), 0) + 1

                added_set = set()

                for id in ids:
                    cat = id.split('/')[0:4]
                    for i in xrange(4):
                        if cat[i] in added_set:
                            continue
                        added_set.add(cat[i])
                        count_id[i][cat[i]] = count_id[i].get(cat[i], 0) + 1
                        count_id_gender[i][cat[i]] = count_id_gender[i].get(cat[i], {})
                        count_id_gender[i][cat[i]][label] = count_id_gender[i][cat[i]].get(label, 0) + 1



        for id, fre in sorted(count_id[selected_id].items(), key=lambda x: x[1]):
            female = count_id_gender[selected_id][id].get(0, 0)
            male = count_id_gender[selected_id][id].get(1, 0)
            total = female + male
            assert total == count_id[selected_id][id]
            portion = female * 1.0 / (female + male)
            if total >= 10:
                if portion > 0.80 or portion < 0.2:
                    print id, total, portion

if __name__ == '__main__':
    da = DataAnalysis()
    # for id in xrange(4):
    da._loadTrainingData(0)