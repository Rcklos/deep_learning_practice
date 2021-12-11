from fer2013 import DataSet
import numpy as np

img_size = 48 * 48 * 1

def fetch_fer2013(x_num = 5, t_num = 5):
    dataset_obj = DataSet(trans_num = x_num, test_num = t_num)
    train_imgs, train_labels, test_imgs, test_labels = dataset_obj.get_data_set()

    dataset = {}
    # img
    dataset['train_img'] = np.array(train_imgs).reshape(-1, img_size)
    dataset['test_img'] = np.array(test_imgs).reshape(-1, img_size)
    
    # label
    dataset['train_label'] = np.eye(10)[train_labels]
    dataset['test_label'] = np.eye(10)[test_labels]

    return dataset


if __name__ == '__main__':
    dataset = fetch_fer2013()
    print(dataset['train_label'])
    print(dataset['test_label'])