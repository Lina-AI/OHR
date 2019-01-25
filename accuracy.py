
def remove_blanks(arr):
    ans = []
    for i in range(len(arr)):
        cur = arr[i]
        while cur and cur[-1] == "":
            cur = cur[:-1]
        ans.append(cur)
    return ans

lookup = {}
def memoize(func):
    global lookup
    def memoized_func(*args):
        global lookup
        if args in lookup.keys():
            return lookup[args]
        ans = func(*args)
        lookup[args] = ans
        return ans
    return memoized_func

@memoize
def lcs(string1, string2):
    if not string1 or not string2:
        return 0
    if string1[0] == string2[0]:
        return 1 + lcs(string1[1:], string2[1:])
    lcs1 = lcs(string1[1:], string2)
    lcs2 = lcs(string1, string2[1:])
    return max([lcs(string1, string2[1:]),lcs(string1[1:], string2)])

def load_file(file_path):
    f = open(file_path)
    contents = f.read()
    f.close()
    contents = contents.replace("' '", "','")
    contents = contents.replace("'\n", "',\n")
    contents = contents.replace("' \"", "',\"")
    contents = contents.replace("\" '", "\",'")
    contents = contents.replace("[", "(")
    contents = contents.replace("]", "),")
    contents = contents[:-1]
    contents = "(" + contents + ")"
    return eval(contents)


labels = load_file("target.txt")
predicted = load_file("output.txt")

labels = remove_blanks(labels)
predicted = remove_blanks(predicted)
print(len(labels), len(predicted))

use_lcs =True
if not use_lcs:
    total = 0
    correct = 0
    for i in range(len(labels)):
        if i ==500000:
            print(labels[i], "\n", predicted[i])
            quit()
        for j in range(len(labels[i])):
            try:
                if labels[i][j] == predicted[i][j]:
                    correct +=1
            except IndexError:
                pass
            total +=1

    print("%.3f"%(100*correct/total))

if use_lcs:
    performance = []
    for i in range(20):
        lookup = {}
        if labels[i]:
            this_accuracy = lcs(labels[i], predicted[i]) / len(labels[i])
            performance.append(this_accuracy)
        else:
            print("empty")
    avg = sum(performance)/len(performance)
    print("%.3f"%(100*avg))
