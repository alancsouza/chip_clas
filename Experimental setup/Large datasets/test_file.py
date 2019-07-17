f = open("names.txt", "a+")

names = ["alan", "ricardo", "Júlia", "Breno"]

for name in names:
    
    f.write("nome: %s \n" % name)
    
f.close()

