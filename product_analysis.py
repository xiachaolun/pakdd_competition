def _loadData(file):
    products_seq = []
    with open(file) as f:
        for line in f.readlines():
            d = line.strip().split(',')
            products_str = d[3]
            ids = products_str.split(';')
            products = []
            for product in ids:
                ids = product.split('/')[0:4]
                products.append((ids[0], ids[1], ids[2], ids[3]))
            products_seq.append(products)
    return products_seq

def findCommonFeatures(level=3):
    train_products = _loadData('data/trainingData.csv')

    training_product_set = set()
    for product_record in train_products:
        for product in product_record:
            name = product[level]
            training_product_set.add(name)


    test_products = _loadData('data/testData.csv')
    test_product_set = set()
    for product_record in test_products:
        for product in product_record:
            name = product[level]
            test_product_set.add(name)

    # how in training but not in test
    valid_names = set()
    for name in training_product_set:
        if name in test_product_set:
            valid_names.add(name)
    valid_name_index = {}

    valid_name_list = sorted(list(valid_names))
    for i in xrange(len(valid_name_list)):
        valid_name_index[valid_name_list[i]] = i
    return valid_names, valid_name_index

if __name__ == '__main__':
    findCommonFeatures(1)