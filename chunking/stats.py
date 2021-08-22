# Let's use train.label.txt

train_labeltxt = open('./data/train.label.txt', "r")

train_label = train_labeltxt.read()

train_labeltxt.close()

examples_labels_list = train_label.splitlines()

# list containing lists with the label of each input train example
train_list_of_lists = [example.split() for example in examples_labels_list] 


# Let's use test.label.txt

test_labeltxt = open('./data/test.label.txt', "r")

test_label = test_labeltxt.read()

test_labeltxt.close()

test_examples_labels_list = test_label.splitlines()

# list containing lists with the label of each input test example
test_list_of_lists = [example.split() for example in test_examples_labels_list]



def label_parts(label):  # function to get the BIO tag and constituent tags separately
    
    label_BIO = label[0]
    label_POS = label[2:]
    return label_BIO, label_POS


def contraints_counters(lista):  # counting statistics for constraints for one input sequence
    
    counter_c1 = 0   # counter for constraint I follows O
    counter_c2 = 0   # counter for constraint I of a class follows B of a different class
    counter_c3 = 0   # counter for B-VP
    b_vp = False # Flag for the presence of B-VP in the input sentence
    
    for i in range(len(lista)-1):
        
        if label_parts(lista[i])[0]=='O' and label_parts(lista[i+1])[0]=='I':
            #print('EJE')
            counter_c1+=1
        if label_parts(lista[i])[0]=='B' and label_parts(lista[i+1])[0]=='I' and label_parts(lista[i])[1]!=label_parts(lista[i+1])[1]:
            #print('UPA')
            counter_c2+=1
        if label_parts(lista[i])[0]=='B'  and label_parts(lista[i])[1]=='VP':
            #print('AJA')
            counter_c3+=1
    
    if label_parts(lista[len(lista)-1])[0]=='B' and label_parts(lista[len(lista)-1])[1]=='VP':
        counter_c3+=1

    if counter_c3!=0:
        b_vp = True

    return counter_c1, counter_c2, counter_c3, b_vp


# TRAIN SET SIZE 8936
#print(len(train_list_of_lists))
# TEST SET SIZE 2012
#print(len(test_list_of_lists))

def constraints_stats(list_of_lists): # stats of constraints for whole data
    
    total_c1=0
    total_c2=0
    total_c3=0
    total_b_vp_not=0
    
    for input_sent in list_of_lists:
        con_1, con_2, con_3, b_vp_is = contraints_counters(input_sent)
        total_c1+=con_1
        total_c2+=con_2
        total_c3+=con_3

        if b_vp_is==False:
            total_b_vp_not+=1
    
    return total_c1, total_c2, total_c3, total_b_vp_not




print(constraints_stats(train_list_of_lists))
# Number of input train sentences that do not contain B-VP tag: 293
print(constraints_stats(test_list_of_lists))
# Number of input test sentences that do not contain B-VP tag: 71













