#coding:utf-8
import re, collections

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    #使用defaultdict的好处在于当访问一个不存在的键值的时候会调用入参函数，并将结果作为这个key的value
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(file('big.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

#实现编辑距离为1的操作,也就是说只改变一个字母
def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   #删除一个字母
   deletes    = [a + b[1:] for a, b in splits if b]
   #移动一个字母
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   #替换一个字母
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   #插入一个字母
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

#实现编辑距离为2的操作,也就是说只改变二个字母
def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

if __name__ == "__main__":
    word = "name1"
    print correct(word)