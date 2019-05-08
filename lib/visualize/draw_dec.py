import os
import cv2
import json
import pickle
from multiprocessing import Pool
import numpy as np
from graphviz import Digraph

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

def get_raw(result_file):
    if not os.path.isfile(result_file):
        print('no such file !!!')
        return {}
    with open(result_file, 'rb') as fin:
        result = pickle.load(fin)
    return result

def get_result(raw):
    with open('pretrained/objects.json', 'r') as fin:
        obj_names = json.load(fin)
    with open('pretrained/predicates.json', 'r') as fin:
        rel_names = json.load(fin)
    topk = 16
    pairs = [[obj([0, 0, 0, 0], '0'), '', obj([0, 0, 0, 0], '1'), 0] for _ in range(topk)]
    for i in range(topk):
        pairs[i][0].bbox, pairs[i][0].label = raw['boxes_s_top'][i], obj_names[raw['labels_s_top'][i]]
        pairs[i][1] = rel_names[raw['labels_p_top'][i]]
        pairs[i][2].bbox, pairs[i][2].label = raw['boxes_o_top'][i], obj_names[raw['labels_o_top'][i]]
        pairs[i][3] = raw['scores_top'][i]

    objs = []
    sbjs = []

    for pair in pairs:
        if not pair[0] in objs:
            objs.append(pair[0])
            objc = objs[-1]
        else:
            objc = objs[objs.index(pair[0])]
        #objc.rel = objc.rel + [prd(pair[1], pair[2], pair[3])]
        objc.add_rel(pair[1], pair[2], pair[3])
        del objc
    pairs[0]
    for pair in pairs:
        if not (pair[2] in objs or pair[2] in sbjs):
            sbjs.append(pair[2]) 
    for i in range(len(objs)):
        objs[i].id = '__s{}'.format(i)
    for i in range(len(sbjs)):
        sbjs[i].id = '__o{}'.format(i)
    return objs, sbjs

def draw_image(img_file, objs, sbjs, output_file, 
               show_words = False, show_line = False, view = False):
    def _draw_box(im, obj2, color = (255, 0, 0)):
        return cv2.rectangle(im,
                             (int(obj2.bbox[0]), int(obj2.bbox[1])),
                             (int(obj2.bbox[2]), int(obj2.bbox[3])),
                             (255, 0, 0), 2)
    if not os.path.isfile(img_file):
        print('image not found !!!')
        return None
    im_out = cv2.imread(img_file)
    for x in objs + sbjs:
        im_out = _draw_box(im_out, x)
    if show_words:
        img = im_out.astype(np.uint8)
        for x in objs + sbjs:
            x0, y0 = int(x.bbox[0]), int(x.bbox[1] + 12)
            scale = 0.55
            # Compute text size.
            txt = x.label
            font = cv2.FONT_HERSHEY_SIMPLEX
            ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, scale, 1)
            # Place text background.
            back_tl = x0, y0 - int(1.3 * txt_h)
            back_br = x0 + txt_w, y0
            cv2.rectangle(img, back_tl, back_br, (18, 127, 15), -1)
            # Show text.
            txt_tl = x0, y0 - int(0.3 * txt_h)
            cv2.putText(img, txt, txt_tl, font, scale, (218, 227, 218), lineType=cv2.LINE_AA)
        im_out = img
    cv2.imwrite(output_file, im_out)
    return im_out

def draw_sg(objs, sbjs, output_file, scale):
    g = Digraph('image')
    g.attr('graph', size="7, 7")
    for obj in objs + sbjs:
        g.node(obj.id, label = obj.label)

    edges = []
    for obj in objs:
        for x in obj.rel:
            '''
            if not x.obj2 in objs:
                g.node(x.obj2.id, label = x.obj2.label + x.obj2.id)
            '''
            for y in objs + sbjs:
                if y == x.obj2:
                    if not edg(obj.id, y.id, x.rel) in edges:
                        g.edge(obj.id, y.id, x.rel, penwidth = str(scale * (x.score ** 3)))
                        edges = edges + [edg(obj.id, y.id, x.rel)]
                    break

    g.render(output_file, view = False)
    return

def img_and_sg(raw_mini):
    objs, sbjs = get_result(raw_mini)
    img_file = raw_mini['image']
    max_score = raw_mini['scores_top'][0]
    scale = 4.0 / (max_score ** 3)
    output_file = img_file.replace('.jpg', '') + '-out'
    draw_image(img_file, objs, sbjs, output_file + '.jpg', show_words = True)
    draw_sg(objs, sbjs, output_file + '.gv', scale)
    with open(output_file + '.pkl', 'wb') as fout:
        pickle.dump({"objs": objs, "sbjs": sbjs}, fout)
    return

if __name__ == '__main__':
    #print(get_raw('test-out.pkl'))
    '''
    raw = get_raw('test-out.pkl')[0]
    objs, sbjs = get_result(raw)
    print(objs)
    print(sbjs)
    draw_image(raw['image'], objs, sbjs, 'test-out.jpg', show_words = True)
    draw_sg(objs, 'test-out.gv')
    '''
    p = Pool(8)
    raw = get_raw('sg-out.pkl')[:200]
    for i in raw:
        p.apply_async(img_and_sg, (i, ))
    p.close()
    p.join()
    '''
    raw = get_raw('test-out.pkl')[0]
    img_and_sg(raw)
    '''
