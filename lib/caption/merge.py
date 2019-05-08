
import nltk
import json
import os
from caption.wordlist import *

class prd():
    def __init__(self, rel, obj2, score):
        self.rel = rel
        self.obj2 = obj2
        self.score = score
    def __str__(self):
        return str({'rel':self.rel, 'obj2':self.obj2.label + self.obj2.id})
    def __repr__(self):
        return str(self)
    def __eq__(self, prd2):
        if self.rel == prd2.rel and self.obj2 == prd2.obj2:
            return True
        return False
    def __lt__(self, prd2):
        if self.score < prd2.score:
            return True
        return False

class obj():
    def __init__(self, bbox, label):
        obj.bbox = bbox
        obj.label = label
        obj.display = ''
        obj.id = ''
        obj.rel = []
    def __str__(self):
        out_dict = {}
        out_dict['label'] = self.label
        out_dict['id'] = self.id
        out_dict['rels'] = self.rel[:5]
        return str(out_dict)
    def __repr__(self):
        return str(self) + '\n'
    def __len__(self):
        if self.rel == []: return 0
        return len(self.rel)
    def __eq__(self, obj2):
        if self.label != obj2.label: return False
        minbox = [max(self.bbox[0], obj2.bbox[0]),
                  max(self.bbox[1], obj2.bbox[1]),
                  min(self.bbox[2], obj2.bbox[2]),
                  min(self.bbox[3], obj2.bbox[3])]
        if minbox[2] < minbox[0] or minbox[3] < minbox[1]: return False
        maxbox = [min(self.bbox[0], obj2.bbox[0]),
                  min(self.bbox[1], obj2.bbox[1]),
                  max(self.bbox[2], obj2.bbox[2]),
                  max(self.bbox[3], obj2.bbox[3])]
        if 0.4 > (((minbox[2] - minbox[0]) *  (minbox[3] - minbox[1])) / 
                   ((maxbox[2] - maxbox[0]) * (maxbox[3] - maxbox[1]))):
            return False
        return True
    def __contains__(self, rel_mini):
        for x in self.rel:
            if rel_mini[0] == x.rel and rel_mini[1] == x.obj2:
                return True
        return False
    def add_rel(self, rel, obj2, score):
        if [rel, obj2] in self: return
        #self.rel.append(prd(rel, obj2, score))
        self.rel = self.rel + [prd(rel, obj2, score)]
        return
    def sort(self):
        self.rel.sort()
        return 

class objlist():
    def __init__(self):
        self.objs = []
    def append(self, label):
        if not (label in self.objs or get_pl(label) in self.objs):
            self.objs.append(label)
        else:
            if not get_pl(label) in self.objs:
                self.objs[self.objs.index(label)] = get_pl(label)
    def __len__(self):
        return len(self.objs)
    def __str__(self):
        if 0 == len(self): return 'xxxxxx'
        if 1 == len(self): return get_gc(self.objs[0]) + self.objs[0]
        if 2 == len(self): return '{}{} and {}{}'.format(
            get_gc(self.objs[0]), self.objs[0], get_gc(self.objs[1]), self.objs[1])
        out = ''
        for i in range(len(self) - 1):
            out = out + get_gc(self.objs[i]) + self.objs[i] + ', '
        out = out + 'and {}{}'.format(get_gc(self.objs[-1]), self.objs[-1])
        return out
    def __repr__(self):
        return str(self)
    def isempty(self):
        return len(self) == 0

class edg():
    def __init__(self, sbj_id, obj_id, rel):
        self.sbj_id = sbj_id
        self.obj_id = obj_id
        self.rel = rel
    def __str__(self):
        return str({"sbj": self.sbj_id, "obj": self.obj_id, "rel": self.rel})
    def __repr__(self):
        return str(self) + '\n'
    def __eq__(self, rel2):
        if self.sbj_id == rel2.sbj_id and self.obj_id == rel2.obj_id and self.rel == rel2.rel:
            return True
        return False

def get_result(raw):
    with open('pretrained/objects.json', 'r') as fin:
        obj_names = json.load(fin)
    with open('pretrained/predicates.json', 'r') as fin:
        rel_names = json.load(fin)
    topk = 12
    pairs = [[obj([0, 0, 0, 0], '0'), '', obj([0, 0, 0, 0], '1'), 0] for _ in range(topk)]
    for i in range(topk):
        pairs[i][0].bbox, pairs[i][0].label = raw['boxes_s_top'][i], obj_names[raw['labels_s_top'][i]]
        pairs[i][1] = rel_names[raw['labels_p_top'][i]]
        pairs[i][2].bbox, pairs[i][2].label = raw['boxes_o_top'][i], obj_names[raw['labels_o_top'][i]]
        pairs[i][3] = raw['scores_top'][i]

    objs = []

    for pair in pairs:
        if not pair[0] in objs:
            objs.append(pair[0])
            objc = objs[-1]
        else:
            objc = objs[objs.index(pair[0])]
        #objc.rel = objc.rel + [prd(pair[1], pair[2], pair[3])]
        objc.add_rel(pair[1], pair[2], pair[3])
        del objc
        if not pair[2] in objs:
            objs.append(pair[2])
            objc = objs[-1]
        else:
            objc = objs[objs.index(pair[2])]
        objc.add_rel(get_op(pair[1]), pair[0], pair[3])
        del objc
    for i in range(len(objs)):
        objs[i].id = '__s{}'.format(i)
    return objs

def issame(wd1, wd2):
    if wd1 == wd2: return True
    if wd1 == get_pl(wd2): return True
    if wd2 == get_pl(wd1): return True
    if wd1 in get_sy(wd2): return True
    if wd2 in get_sy(wd1): return True 
    return False

def wrap(base, word, rel, objl):
    if 'xxx' in rel:
        add = rel.replace('xxx', str(objl))
    else:
        add = '{} {}'.format(rel, str(objl))
    return base.replace(word, word + ' ' + add)

def get_merged(base, raw):
    objs = get_result(raw)
    words = nltk.pos_tag(nltk.word_tokenize(base.replace('<start>', '').replace('<end>', '')))
    pot = [word[0] for word in words if 'NN' == word[1] or 'NNS' == word[1]]
    vbs = [word[0] for word in words if 'VBG' == word[1]]
    if len(vbs) >= 1:
        vb = vbs[0]
        vbp = base.find(vb)
        words_trim = nltk.pos_tag(nltk.word_tokenize(base[:vbp].replace('<start>', '').replace('<end>', '')))
        sbj = ['', '']
        for word in words_trim:
            if word[1] == 'NN' or word[1] == 'NNS':
                sbj = word
        if sbj[1] == 'NN': base = base.replace(vb, 'is ' + vb)
        else: base = base.replace(vb, 'are' + vb)

    targets = []
    for t in pot:
        for objc in objs:
            if issame(objc.label, t):
                targets = targets + [(objc, t)]
        del objc
    
    for target in targets:
        objt = target[0]
        word = target[1]
        objt.sort()
        cnt = 0
        objd = [[], []]
        reld = ['', '']
        for i in range(len(objt)):
            if cnt > 1: break
            relc = objt.rel[i].rel
            if relc in reld: continue
            if 'has' == relc or 'of' == relc: continue
            objl = objlist()
            if not objt.rel[i].obj2.label in base:
                objl.append(objt.rel[i].obj2.label)
            for j in range(i + 1, len(objt)):
                if objt.rel[j].rel == relc:
                    if objt.rel[j].obj2.label in base:
                        continue
                    if get_pl(objt.rel[j].obj2.label) in base: 
                        continue
                    objl.append(objt.rel[j].obj2.label)
            if not objl.isempty():
                objd[cnt] = objl
                reld[cnt] = relc
                cnt += 1
            del objl
        if not [] == objd[1]:
            base = wrap(base, word, reld[1], objd[1])
        if not [] == objd[0]:
            base = wrap(base, word, reld[0], objd[0])
    
    
    if not '.' in base:
        base = base + '.'
    return base