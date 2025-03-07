dictionary = {}
file = open('song.txt')

for line in file:        #It
    line = line.rstrip() #just
    words = line.split() #works.
    for word in words:
        clean_word = word.rstrip(",").lower()
        if clean_word not in dictionary:
            dictionary[clean_word] = 1
        else:
            dictionary[clean_word] += 1


file.close()

unique_word_count = 0
print(dictionary)
for word, number in dictionary.items():
    if number == 1:
        unique_word_count += 1
        print(word)
print(unique_word_count)