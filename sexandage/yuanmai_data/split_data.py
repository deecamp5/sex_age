from  sklearn.model_selection import train_test_split
mylabel = open("./data.txt").readlines()
train, test = train_test_split(mylabel, test_size=0.2)
print(len(train))
fp = open("train.txt", "w")
for train_example in train:
    fp.write(train_example)
fp.close()

fp = open("test.txt", "w")
for test_example in test:
    fp.write(test_example)
fp.close()
print(len(test))