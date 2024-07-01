import matplotlib.pyplot as plt
import sys

indices = []
times = []
file = open(sys.argv[1], "r")
counter = 1
for line in file:
  if counter % 2:
    print("adding index ")
    print(line)
    indices.append(int(line))
  else:
    print("adding time ")
    print(line)
    times.append(int(line))
  counter += 1

print("have indices and times:")
print(len(indices))
print(len(times))

plt.figure(figsize=(10,10))
plt.scatter(indices, times)
plt.xlabel('indices')
plt.ylabel('access times [ms]')

ax = plt.gca()
ax.set_xlim([min(indices), max(indices)+1])
ax.set_ylim([min(times), max(times)+1])


filename, _ = sys.argv[1].split(".")
filename = filename + ".png"
plt.savefig(filename)
#plt.show()