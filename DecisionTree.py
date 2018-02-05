from math import log2


category_dict = dict()
label = None
dataset = []
Y = None

def create_category_dict_():
    i = 0
    for s in zip(*dataset):
        category_dict[label[i]] = set(s)
        i += 1

def read_data(file_name):
    with open(file_name, 'r') as fs:
        for i, f in enumerate(fs):
            if i == 0:
                global label, category_dict, Y
                label = f.strip().split(',')
                Y = f.strip().split(',')[-1]
            else:
                dataset.append(f.strip().split(','))

    create_category_dict_()

read_data('fishing.data')

def first_entropy():
    y = list(zip(*dataset))[-1]
    y = [y.count(y_cate) for y_cate in category_dict[Y]]
    y = list(map(lambda x: -x/sum(y)*log2(x/sum(y)), y))
    return sum(y)



def entropy(s, attr_cate_tuple):
    attribute, cate = attr_cate_tuple
    idx = label.index(attribute)

    cnt = 0
    rst = 0
    l = []
    for val in category_dict[Y]:
        for i in s:
            if dataset[i][idx] == cate:
                if dataset[i][-1] == val:
                    cnt += 1
        l.append(cnt)
        cnt = 0

    for i in l:
        if i != 0:
            rst += (-i / sum(l))*log2(i / sum(l))

    return (rst, sum(l))


def gain(prev_gain, s, attr):
    entropy_list = []
    category = category_dict[attr]
    total = len(s)
    rst = prev_gain
    for cate in category:
        entro, sum_ = entropy(s, (attr, cate))
        entropy_list.append({cate : entro})
        rst += -(sum_/total) * entro

    return (rst, entropy_list)


def findTheBestFeature(s, attributes, entropy):
    max_, entropy_list = gain(entropy, s, attributes[0])
    max_gain_attr = attributes[0]

    for attr in attributes[1:]:
        temp, entropy_l = gain(entropy, s, attr)
        if temp > max_:
            max_ = temp
            max_gain_attr = attr
            entropy_list = entropy_l
    return (max_gain_attr, entropy_list)

def tree(s, attributes, entro=first_entropy()):

    # recursion stop condition
    classLabel = [dataset[i][-1] for i in s]
    temp = classLabel[0]
    classLabel = set(classLabel)
    if len(classLabel) == 1:
        return temp
    if not attributes:
        return 0

    # find the best feature (highest gain)
    best_feature, entro_list = findTheBestFeature(s, attributes, entro)
    idx = label.index(best_feature)
    attributes.remove(best_feature)

    model = {best_feature : {}}

    for dic in entro_list:
        for cate in dic:
            new_set = []
            for i in s:
                if dataset[i][idx] == cate:
                    new_set.append(i)
            temp_attr = attributes[:]
            model[best_feature][cate] = tree(new_set, temp_attr, dic[cate])
    return model


def make_prediction(model, testingset):

    rst = model
    while isinstance(rst, dict):
        key = list(rst.keys())[0]
        idx = label.index(key)
        rst = rst[key][testingset[idx]]

    return rst



dtree = tree([i for i in range(len(dataset))], label[:-1])
print(dtree)
for i in dataset:
    print(make_prediction(dtree, i))