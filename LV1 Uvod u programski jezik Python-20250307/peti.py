spam = 0
spamwords = 0
ham = 0
endwith = 0
hamwords = 0

file = open("SMSSpamCollection.txt")


for line in file :

    line = line.rstrip()
    words = line.split("\t")
    wordsline = words[1].split()
    if words[0] == "spam":
        spam += 1
        spamwords += len(wordsline)
        if words[1].endswith('!'):
            endwith += 1
    if words[0] == "ham":
        ham += 1
        hamwords += len(wordsline)

file.close()

print(float(spamwords)/float(spam))
print(float(hamwords)/float(ham))
print(endwith)
