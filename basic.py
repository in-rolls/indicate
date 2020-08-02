# %%
def nagari_slicer(word):
    vowels = ["ा", "ि", "ी", "ु", "ू", "ृ", "े", "ै", "ो", "ौ"]
    chars = []
    for c in word:
        if c in vowels:
            chars[-1] = chars[-1] + c
        else:
            chars.append(c)
    return chars
# %%
nagari_slicer("चौकिदार")
char_xw = {'चौ': "chau",
           'कि': "ki",
           'दा': "da",
           'र': "r"}
''.join([char_xw[c] for c in nagari_slicer("चौकिदार")])
# %%
b = 'क्षेत्र पंचायत प्रमुख'

b[0]
b[1]
b[2]
b[3]
# %% todo
# populate dict with transliterations from letter_list.txt
# then read in dict to populate char_xw

# file = open("dict_string.txt", "r")
# contents = file.read()
# char_xw = ast.literal_eval(contents)
