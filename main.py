# file to test code
import dataset

if __name__ == "__main__":
    train_set = dataset.QuickDrawDataset(root="/data", train=True, download=True)

    print(train_set)

    # labels = dataset.QuickDrawDataset.categories()

    # print(labels)
