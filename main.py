# file to test code
import dataset

if __name__ == "__main__":
    # train_set = dataset.QuickDrawDataset(root="/data", train=True, download=False)

    # train_points = train_set.get()

    # print(train_points)

    labels = dataset.QuickDrawDataset.categories()

    print(labels)
