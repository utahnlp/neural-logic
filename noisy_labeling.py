import torch
import argparse
from config_ijcai import *
from utils import setseed
from model import OpeModuloModel, Model


## we create a torch data class of the pair data now with the noisy labels
class noisy_labeled_pairset(torch.utils.data.Dataset):

    def __init__(self, pair_data, noisy_label):

        self.data = pair_data

        self.noisy_sum_labels, self.noisy_prod_labels  = noisy_label
    
        self.noisylabeledset = []

        if len(pair_data)!=len(self.noisy_sum_labels):
            print('ERROR')
        
        for i in range(len(self.data)):
            labeled_im1 = self.data[i][0]
            labeled_im2 = self.data[i][1]
            catim = self.data[i][2]
            noisy_labeled_example = (labeled_im1, labeled_im2, catim, self.noisy_sum_labels[i], self.noisy_prod_labels[i])
            self.noisylabeledset.append(noisy_labeled_example)

    def __len__(self):
        return len(self.noisylabeledset)

    def __getitem__(self, index):

       # "returns : (image_2,label_2), (image_1,label_1), concat_image, noisy_label_sum, *label_prod*)"

        o1 = self.noisylabeledset[index][0]
        o2 = self.noisylabeledset[index][1]
        o3 = self.noisylabeledset[index][2]
        o4 = self.noisylabeledset[index][3]
        o5 = self.noisylabeledset[index][4]
            
        return o1, o2, o3, o4, o5


# Code to label the pair data with a digit classifier, for a pair we predict the label for each image in the pair, then we add and multiply those prediction in modulo 10 and append those two labels to the pair. 
def create_noisy_labeled_pair_dataset(digit_size=5000, pair_size=5000, dev_test_size=50000, option=1, learning_rate=0.001, optimizer=0, tnorm='prod', model_seed=20, operator_data_seed=20, n_epochs=500):
    path_to_pretrained_digit_model = MODEL_DIR + '/' + SINGLE_MODEL_NAME.format(digit_size, learning_rate, optimizer, tnorm, model_seed, n_epochs)
    path_to_pair_unlabeled_data = DATA_DIR + '/' + OPERATOR_MNIST_FILE.format(option, digit_size, pair_size, dev_test_size, operator_data_seed)

    # this function uses the same seed that was used  to create the DIGIT and PAIR datasets to create the NOISY labeled dataset   
    setseed(operator_data_seed)
    pair_train, operator_val, operator_test = torch.load(path_to_pair_unlabeled_data)
    f_0 = Model()
    f_0.load_state_dict(torch.load(path_to_pretrained_digit_model))
    f_0.cuda()
    f_0.eval()

    pair_train_dataloader = torch.utils.data.DataLoader(pair_train, batch_size=64, shuffle=False, num_workers = 0)

    # we are going to fill a tensor init_tensor  with the predicted labels for each unlabeled example provided by digit f_0 model.
    init_tensor = torch.tensor([], dtype=torch.int64).cuda()
    init_tensor_mult = torch.tensor([], dtype=torch.int64).cuda()
    for batch_pair in pair_train_dataloader: 

        img1_labeledpair, img2_labeledpair, cat_img, _, _ = batch_pair

        img1 = img1_labeledpair[0]
        img2 = img2_labeledpair[0]

        if (torch.cuda.is_available()):
            img1 = img1.cuda()
            img2 = img2.cuda()
            cat_img = cat_img.cuda()

    # logits for right and left images in the operation
        logits_img1 = f_0(img1)
        logits_img2 = f_0(img2)
    # probabilities for each label for im1 and im2    
        img1_probs = torch.nn.functional.softmax(logits_img1, dim=1)
        img2_probs = torch.nn.functional.softmax(logits_img2, dim=1)
    # predicted label for im1 im2 according to the highest probability for a label
        _, pred_labels_img1 = torch.max(img1_probs, dim=1)
        _, pred_labels_img2 = torch.max(img2_probs, dim=1)
    # we add (mult) up predictions for im1 and im2 (given by f_0) modulo 10. That will be the (noisy) labels for cat_img (of im1 im2)
        sum_noisy_label = (pred_labels_img1+pred_labels_img2)%10
        prod_noisy_label = (pred_labels_img1*pred_labels_img2)%10

        init_tensor = torch.cat((init_tensor,sum_noisy_label),0)
        init_tensor_mult = torch.cat((init_tensor_mult,prod_noisy_label),0)
    print(init_tensor)
    print(init_tensor_mult)
    noisy_labeled_pairdata = noisy_labeled_pairset(pair_train,(init_tensor, init_tensor_mult))
    torch.save(noisy_labeled_pairdata, DATA_DIR+'/'+NOISY_LABELED_PAIR.format(option,pair_size,operator_data_seed,digit_size,learning_rate, optimizer, tnorm, model_seed, n_epochs))







if __name__=="__main__":

    parser= argparse.ArgumentParser()

    parser.add_argument('--digit_size', type=int, default=5000, help="DIGIT data size")
    parser.add_argument('--pair_size', type=int, default=5000, help="PAIR data size") 
    parser.add_argument('--dev_test_size', type=int, default=50000, help="PairDEV and PairTEST sets sizes")
    parser.add_argument('--option', type=int, default=2, help="Dataset setting/option for training operator models. Can be one of (1,2,3)")
    parser.add_argument('--ope_seed', type=int, default=20, help="Seed used to create the PAIR data to be labeled")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate used to train the Digit model that will be used for labeling')
    parser.add_argument('--optimizer', type=int, default=1, help='1 Adam, 0 SGD. Optimizer used to train the Digit model that will be used for labeling')
    parser.add_argument('--tnorm', type=str, default='prod', help="t-norm used to train the Digit model that will be used for labeling")
    parser.add_argument('--model_seed', type=int, default=20, help="seed used to train de Digit model that will be used for labeling")
    parser.add_argument('--nepochs', type=int, default=1600, help="Number of epochs use to train the Digit model that will be used for labeling")

    args = parser.parse_args()



# we have to input the information of the digit model we want to use to labeled the data
    create_noisy_labeled_pair_dataset(digit_size=args.digit_size, pair_size=args.pair_size, dev_test_size=args.dev_test_size, option=args.option, learning_rate=args.learning_rate, 
            optimizer=args.optimizer, tnorm=args.tnorm, model_seed=args.model_seed, operator_data_seed=args.ope_seed, n_epochs=args.nepochs)





