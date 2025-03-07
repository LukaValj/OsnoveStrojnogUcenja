




list = []
while 1:
    try:
        x = input("Upisi broj:")
        if x == "Done":
            break
        else:
            list.append(float(x))
    except ValueError:
        print("Nije broj, probaj opet.")

length = len(list)
min = min(list)
max = max(list)
avg = sum(list) / len(list)

print(f"Broj brojeva: {length}")
print(f"Minimumj: {min}")
print(f"Maksimum: {max}")
print(f"Sredina: {avg}")

list.sort()
print(list)