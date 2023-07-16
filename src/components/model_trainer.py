import os
a = os.path.join("hello","world")
b = os.path.dirname(a)
print(b)
os.makedirs(b)