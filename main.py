# file to test code
import dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_set = dataset.QuickDrawDataset(root="/data", train=True, download=False)

    print(train_set)

    # print(len(train_set))

    # for x, y in train_set:
    #     img = x.numpy()

    #     print(y)

    #     plt.imshow(img, cmap="gray")
    #     plt.show()
    # break

    # labels = dataset.QuickDrawDataset.categories()

    # print(labels)
