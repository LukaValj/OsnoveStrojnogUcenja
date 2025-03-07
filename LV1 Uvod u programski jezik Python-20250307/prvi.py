print("Radni sati:")
x = float(input())
print("\n")
print("eura/h:")
y = float(input())
print("\n")

def total_euro(sati, satnica):
    placa = sati * satnica
    return placa

ukupno = total_euro(x,y)
print("Ukupno:")
print(ukupno)