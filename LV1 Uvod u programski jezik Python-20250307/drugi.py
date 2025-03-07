def f():
    try:
        print("Ocijena:")
        x = float(input())
        print("\n")
        if (x<0 or x>1):
            print("Not acceptable, try again.")
            quit()
        
        if x >= 0.9:
                return 'A'
        elif x >= 0.8:
                return 'B'
        elif x >= 0.7:
                return 'C'
        elif x >= 0.6:
                return 'D'
        else:
                return 'F'
    except ValueError:
           print("Not a numbery try again.")
           quit()

result = f()
print("Your grade:")
print(result)