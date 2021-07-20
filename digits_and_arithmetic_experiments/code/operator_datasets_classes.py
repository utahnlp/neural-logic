import torch
import random
import numpy as np
from torchvision import transforms, datasets


def data_dictio_by_labels(dataset):
    """
    dictionary that gathers all images of the same label under one same key for the given dataset
    of the form {0:[(imof0, 0), (imof0, 0),...], 1: [...],...}

    """
    dataset_map = {}
    for i in range(len(dataset)):
        label = dataset[i][1]
        if label not in dataset_map:
            dataset_map[label] = []
        dataset_map[label].append((dataset[i][0], dataset[i][1]))
    
    return dataset_map


# Option 1: For every image pair (a, b) the reverse of the same images (b, a) shows up in the training data
class operatorMNIST_1(torch.utils.data.Dataset):

    def __init__(self, mnist_data, size=50000):

        self.data = mnist_data

        self.num_examples = size

        self.set_operator = []

        self.set_included = set()

        sampled = 0
        while sampled < self.num_examples:
            # left and right images of the operation randomly selected
            idx1 = random.randint(0,len(self.data)-1)
            idx2 = random.randint(0,len(self.data)-1)
            while idx2==idx1:
                idx2 = random.randint(0,len(self.data)-1)

            if (idx1,idx2) not in self.set_included:
                # appending the two images
                image_1 = self.data[idx1][0]
                label_1 = self.data[idx1][1]
                image_2 = self.data[idx2][0]
                label_2 = self.data[idx2][1]
                concat_image = torch.cat((image_1, image_2), dim=2)
                concat_image_reverse = torch.cat((image_2, image_1), dim=2)
                label_sum = (label_1 + label_2)%10
                label_prod = (label_1 * label_2)%10
                self.set_operator.append(((image_1,label_1), (image_2,label_2), concat_image, label_sum, label_prod))
                # appending the reverse of the images
                self.set_operator.append(((image_2,label_2), (image_1,label_1), concat_image_reverse, label_sum, label_prod))
                # adding the two images and their reverse to the set_included set so we don't select them again
                self.set_included.add((idx1, idx2))
                self.set_included.add((idx2, idx1))
                sampled +=2



        print('Number of examples randomly sampled : ' + str(len(self.set_operator)))

    def __len__(self):
        return len(self.set_operator)

    def __getitem__(self, index):

        "returns : (image_2,label_2), (image_1,label_1), concat_image_reverse, label_sum, label_prod)"

        o1 = self.set_operator[index][0]
        o2 = self.set_operator[index][1]
        o3 = self.set_operator[index][2]
        o4 = self.set_operator[index][3]
        o5 = self.set_operator[index][4]

        return o1, o2, o3, o4, o5


# Option 2: For any image pair (a, b) the reverse of the same images (b, a) does not show up in the training data
class operatorMNIST_2(torch.utils.data.Dataset):

    def __init__(self, mnist_data, size=50000):

        self.data = mnist_data

        self.num_examples = size

        self.set_operator = []

        self.set_included = set()

        #print('LENDATA', len(self.data))

        #print('NUM EX',self.num_examples)

        sampled = 0
        # Loop over num_examples
        while sampled < self.num_examples:
            # get idx1, idx2
            idx1 = random.randint(0,len(self.data)-1)
            idx2 = random.randint(0,len(self.data)-1)
            while idx2==idx1:
                idx2 = random.randint(0,len(self.data)-1)

            if (idx1,idx2) not in self.set_included:
                # appending the two images
                image_1 = self.data[idx1][0]
                label_1 = self.data[idx1][1]
                image_2 = self.data[idx2][0]
                label_2 = self.data[idx2][1]
                concat_image = torch.cat((image_1, image_2), dim=2)
                label_sum = (label_1 + label_2)%10
                label_prod = (label_1 * label_2)%10
                self.set_operator.append(((image_1,label_1), (image_2,label_2), concat_image, label_sum, label_prod))
                # adding the two images and their reverse to the set_included set so we don't select them again
                self.set_included.add((idx1, idx2))
                self.set_included.add((idx2, idx1))
                sampled +=1

        print('Number of examples randomly sampled : ' + str(len(self.set_operator)))

    def __len__(self):
        return len(self.set_operator)

    def __getitem__(self, index):

        o1 = self.set_operator[index][0]
        o2 = self.set_operator[index][1]
        o3 = self.set_operator[index][2]
        o4 = self.set_operator[index][3]
        o5 = self.set_operator[index][4]
        return o1, o2, o3, o4, o5


# Option 3: For any pair of numbers (a, b), no image for the reverse (b, a) is included in the training set
class operatorMNIST_3(torch.utils.data.Dataset):

    def __init__(self, mnist_data, size=50000):

        self.data = mnist_data

        self.num_examples = size

        self.set_operator = []

        self.set_included = set()

        self.set_digit_included = set()

        #print('LENDATA', len(self.data))

        #print('NUM EX',self.num_examples)

        sampled = 0

        while sampled < self.num_examples:
            # get idx1, idx2
            idx1 = random.randint(0,len(self.data)-1)
            idx2 = random.randint(0,len(self.data)-1)
            while idx2==idx1:
                idx2 = random.randint(0,len(self.data)-1)

            if ((idx1,idx2) not in self.set_included) and ((self.data[idx1][1], self.data[idx2][1]) not in self.set_digit_included):
                # Get corresponding example and do whatever we want
                image_1 = self.data[idx1][0]
                label_1 = self.data[idx1][1]
                image_2 = self.data[idx2][0]
                label_2 = self.data[idx2][1]
                concat_image = torch.cat((image_1, image_2), dim=2)
                label_sum = (label_1 + label_2)%10
                label_prod = (label_1 * label_2)%10
                self.set_operator.append(((image_1,label_1), (image_2,label_2), concat_image, label_sum, label_prod))
                self.set_included.add((idx1,idx2))
                self.set_digit_included.add((self.data[idx2][1], self.data[idx1][1]))

                sampled +=1

        print('Number of examples randomly sampled : ' + str(len(self.set_operator)))

    def __len__(self):
        return len(self.set_operator)

    def __getitem__(self, index):

        o1 = self.set_operator[index][0]
        o2 = self.set_operator[index][1]
        o3 = self.set_operator[index][2]
        o4 = self.set_operator[index][3]
        o5 = self.set_operator[index][4]
        return o1, o2, o3, o4, o5


# this program is used to create validation and test sets for PAIR
class testOpMNIST(torch.utils.data.Dataset):

    def __init__(self, mnist_data, size=50000):

        self.data = mnist_data

        self.num_examples = size

        self.set_operator = []

        self.set_included = set()

        sampled = 0


        while sampled < self.num_examples:
            # get idx1, idx2
            idx1 = random.randint(0,len(self.data)-1)
            idx2 = random.randint(0,len(self.data)-1)
            while idx2==idx1:
                idx2 = random.randint(0,len(self.data)-1)

            if (idx1,idx2) not in self.set_included:
                image_1 = self.data[idx1][0]
                label_1 = self.data[idx1][1]
                image_2 = self.data[idx2][0]
                label_2 = self.data[idx2][1]
                concat_image = torch.cat((image_1, image_2), dim=2)
                label_sum = (label_1 + label_2)%10
                label_prod = (label_1 * label_2)%10
                self.set_operator.append(((image_1,label_1), (image_2,label_2), concat_image, label_sum, label_prod))
                self.set_included.add((idx1, idx2))
                sampled+=1

        print('Number of examples randomly sampled : ' + str(len(self.set_operator)))

    def __len__(self):
        return len(self.set_operator)

    def __getitem__(self, index):

        o1 = self.set_operator[index][0]
        o2 = self.set_operator[index][1]
        o3 = self.set_operator[index][2]
        o4 = self.set_operator[index][3]
        o5 = self.set_operator[index][4]
        return o1, o2, o3, o4, o5



# this functions creates data to test commutativity property
class Commut_MNIST(torch.utils.data.Dataset):

    def __init__(self, mnist_data, size=50000):
        self.data = mnist_data

        self.num_examples = size

        self.set_operator = []

        self.set_included = set()

        sampled = 0

        while sampled < self.num_examples:
            # get idx1, idx2
            idx1 = random.randint(0,len(self.data)-1)
            idx2 = random.randint(0,len(self.data)-1)
            while idx2==idx1:
                 idx2 = random.randint(0,len(self.data)-1)

            if (idx1,idx2) not in self.set_included:
                image_1 = self.data[idx1][0]
                label_1 = self.data[idx1][1]
                image_2 = self.data[idx2][0]
                label_2 = self.data[idx2][1]
                concat_image = torch.cat((image_1, image_2), dim=2)
                concat_image_reverse = torch.cat((image_2,image_1), dim=2)
                self.set_operator.append((image_1, image_2, concat_image, concat_image_reverse))
                self.set_included.add((idx1, idx2))
                self.set_included.add((idx2, idx1))
                sampled+=1

        print('Number of examples randomly sampled : ' + str(len(self.set_operator)))

    def __len__(self):
        return len(self.set_operator)

    def __getitem__(self, index):

        o1 = self.set_operator[index][0]
        o2 = self.set_operator[index][1]
        o3 = self.set_operator[index][2]
        o4 = self.set_operator[index][3]

        return o1, o2, o3, o4




# class to create the data to test associativity, for this we just need three images and their possible combinations and the total
# according to the operation.
class Assoc_MNIST(torch.utils.data.Dataset):

    def __init__(self, mnist_data, size=50000):
        self.data = mnist_data

        self.num_examples = size

        self.set_operator = []

        self.set_included = set()

        sampled = 0

        while sampled < self.num_examples:
            # get idx1, idx2
            idx1 = random.randint(0,len(self.data)-1)
            idx2 = random.randint(0,len(self.data)-1)
            idx3 = random.randint(0,len(self.data)-1)
            while idx2==idx1 or idx1==idx3 or idx2==idx3:
                idx1 = random.randint(0,len(self.data)-1)
                idx2 = random.randint(0,len(self.data)-1)
                idx3 = random.randint(0,len(self.data)-1)

            if (idx1, idx2, idx3) not in self.set_included:
                image_1 = self.data[idx1][0]
                image_2 = self.data[idx2][0]
                image_3 = self.data[idx3][0]

                # first combination im 1 with im 2
                new_concat_im1_im2 = torch.cat((image_1, image_2), dim=2)
                # second combination im 2 with im 3
                new_concat_im2_im3 = torch.cat((image_2, image_3), dim=2)
                self.set_operator.append((image_1, image_2, image_3, new_concat_im1_im2, new_concat_im2_im3))
                self.set_included.add((idx1, idx2, idx3))
                sampled+=1


        print('Number of examples randomly sampled : ' + str(len(self.set_operator)))

    def __len__(self):
        return len(self.set_operator)

    def __getitem__(self, index):
        o1 = self.set_operator[index][0]
        o2 = self.set_operator[index][1]
        o3 = self.set_operator[index][2]
        o4 = self.set_operator[index][3]
        o5 = self.set_operator[index][4]

        return o1, o2, o3, o4, o5


# class to create the data to test distributivity. Creates the pairs according to the definition of the property.
class Dist_MNIST(torch.utils.data.Dataset):

    def __init__(self, mnist_data, size=50000):
        self.data = mnist_data

        self.num_examples = size

        self.set_operator = []

        self.set_included = set()

        sampled = 0

        while sampled < self.num_examples:
            # get idx1, idx2
            idx1 = random.randint(0,len(self.data)-1)
            idx2 = random.randint(0,len(self.data)-1)
            idx3 = random.randint(0,len(self.data)-1)
            while idx2==idx1 or idx1==idx3 or idx2==idx3:
                idx1 = random.randint(0,len(self.data)-1)
                idx2 = random.randint(0,len(self.data)-1)
                idx3 = random.randint(0,len(self.data)-1)

            if (idx1, idx2, idx3) not in self.set_included:
                image_1 = self.data[idx1][0]
                image_2 = self.data[idx2][0]
                image_3 = self.data[idx3][0]

                # Left distributivy concatenations: a*(b+c) = (a*b)+(a*c); a=image_1_dist, b=image_2_dist, c=image_3_dist
                new_concat_image_bplusc = torch.cat((image_2, image_3), dim=2)
                new_concat_image_2_atimesb = torch.cat((image_1, image_2), dim=2)
                new_concat_image_3_atimesc = torch.cat((image_1, image_3), dim=2)

                # Right distributivity concatenations: (b+c)*a = (b*a)+(c*a)
                new_concat_image_4_btimesa = torch.cat((image_2, image_1), dim=2)
                new_concat_image_5_ctimesa = torch.cat((image_3, image_1), dim=2)

                new_example = ((new_concat_image_bplusc, new_concat_image_2_atimesb, new_concat_image_3_atimesc, new_concat_image_4_btimesa, new_concat_image_5_ctimesa,
                                   image_1, image_2, image_3))

                self.set_operator.append(new_example)
                self.set_included.add((idx1, idx2, idx3))
                sampled+=1

        print('Number of examples randomly sampled : ' + str(len(self.set_operator)))

    def __len__(self):
        return len(self.set_operator)

    def __getitem__(self, index):
        o1 = self.set_operator[index][0]
        o2 = self.set_operator[index][1]
        o3 = self.set_operator[index][2]
        o4 = self.set_operator[index][3]
        o5 = self.set_operator[index][4]
        o6 = self.set_operator[index][5]
        o7 = self.set_operator[index][6]
        o8 = self.set_operator[index][7]

        return o1, o2, o3, o4, o5, o6, o7, o8







"""
THIS CODE BELOW IS NOT USED: TODO- REMOVE
"""


# This functions creates the PAIR (also called Operator) datasets

def create_operator_dataset(datapath, option, single_train_size, single_dev_size, ope_train_size, ope_dev_size):
    
    train, dev = torch.load(datapath)

    if option == 1:

        train_ope_dataset = operatorMNIST_1(train, size=ope_train_size)

        dev_ope_dataset = testOpMNIST(dev, size=ope_dev_size)

    if option == 2:

        train_ope_dataset = operatorMNIST_2(train, size=ope_train_size)

        dev_ope_dataset = testOpMNIST(dev, size=ope_dev_size)

    if option == 3:

        train_ope_dataset = operatorMNIST_3(train, size=ope_train_size)

        dev_ope_dataset = testOpMNIST(dev, size=ope_dev_size)


    path = '/uusoc/scratch/bluefish/mattiamg/datasets/operator_MNIST_datasets/option{}/'.format(option)


    torch.save((train_ope_dataset, dev_ope_dataset), path+'operator{}_bl_single_train{}_single_val{}_ope_train{}_ope_val{}.pt'
            .format(option,single_train_size, single_dev_size, ope_train_size, ope_dev_size))

    print('dataset option {} saved'.format(option))

        


