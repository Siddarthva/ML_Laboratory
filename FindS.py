import csv

a = []
with open('enjoysport.csv', 'r') as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)

print("\nOUTPUT:\n")
print(a)

print("\nThe total number of training instances are : ", len(a) - 1)

num_attribute = len(a[0]) - 1

print("\nThe initial hypothesis is : ")
hypothesis = ['0'] * num_attribute
print(hypothesis)

# Find-S Algorithm
for i in range(1, len(a)):
    if a[i][num_attribute] == 'yes':
        for j in range(num_attribute):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'

    print("\nThe hypothesis for the training instance {} is :".format(i))
    print(hypothesis)

print("\nThe Maximally specific hypothesis for the training instance is")
print("is", hypothesis)
