sl = [u'airplane', u'animal', u'arm', u'bag', u'banana',
      u'basket', u'beach', u'bear', u'bed', u'bench', 
      u'bike', u'bird', u'board', u'boat', u'book', 
      u'boot', u'bottle', u'bowl', u'box', u'boy', 
      u'branch', u'building', u'bus', u'cabinet', u'cap', 
      u'car', u'cat', u'chair', u'child', u'clock', 
      u'coat', u'counter', u'cow', u'cup', u'curtain', 
      u'desk', u'dog', u'door', u'drawer', u'ear', 
      u'elephant', u'engine', u'eye', u'face', u'fence', 
      u'finger', u'flag', u'flower', u'food', u'fork', 
      u'fruit', u'giraffe', u'girl', u'glass', u'glove', 
      u'guy', u'hair', u'hand', u'handle', u'hat', 
      u'head', u'helmet', u'hill', u'horse', u'house', 
      u'jacket', u'jean', u'kid', u'kite', u'lady', 
      u'lamp', u'laptop', u'leaf', u'leg', u'letter', 
      u'light', u'logo', u'man', u'man', u'motorcycle', 
      u'mountain', u'mouth', u'neck', u'nose', u'number', 
      u'orange', u'pant', u'paper', u'paw', u'people', 
      u'person', u'phone', u'pillow', u'pizza', u'plane', 
      u'plant', u'plate', u'player', u'pole', u'post', 
      u'pot', u'racket', u'railing', u'rock', u'roof', 
      u'room', u'screen', u'seat', u'sheep', u'shelf', 
      u'shirt', u'shoe', u'short', u'sidewalk', u'sign', 
      u'sink', u'skateboard', u'ski', u'skier', u'sneaker', 
      u'snow', u'sock', u'stand', u'street', u'surfboard', 
      u'table', u'tail', u'tie', u'tile', u'tire', 
      u'toilet', u'towel', u'tower', u'track', u'train', 
      u'tree', u'truck', u'trunk', u'umbrella', u'vase', 
      u'vegetable', u'vehicle', u'wave', u'wheel', u'window', 
      u'windshield', u'wing', u'wire', u'woman', u'zebra']

pl = [u'airplanes', u'animals', u'arms', u'bags', u'bananas',
      u'baskets', u'beaches', u'bears', u'beds', u'benchs', 
      u'bikes', u'birds', u'boards', u'boats', u'books', 
      u'boots', u'bottles', u'bowls', u'boxes', u'boys', 
      u'branchs', u'buildings', u'buses', u'cabinets', u'caps', 
      u'cars', u'cats', u'chairs', u'children', u'clocks', 
      u'coats', u'counters', u'cows', u'cups', u'curtains', 
      u'desks', u'dogs', u'doors', u'drawers', u'ears', 
      u'elephants', u'engines', u'eyes', u'faces', u'fences', 
      u'fingers', u'flags', u'flowers', u'food', u'forks', 
      u'fruits', u'giraffes', u'girls', u'glasses', u'gloves', 
      u'guys', u'hairs', u'hands', u'handles', u'hats', 
      u'heads', u'helmets', u'hills', u'horses', u'houses', 
      u'jackets', u'jeans', u'kids', u'kites', u'ladies', 
      u'lamps', u'laptops', u'leafs', u'legs', u'letters', 
      u'lights', u'logos', u'men', u'men', u'motorcycles', 
      u'mountains', u'mouths', u'necks', u'noses', u'numbers', 
      u'oranges', u'pants', u'paper', u'paws', u'people', 
      u'persons', u'phones', u'pillows', u'pizzas', u'planes', 
      u'plants', u'plates', u'players', u'poles', u'posts', 
      u'pots', u'rackets', u'railings', u'rocks', u'roofs', 
      u'rooms', u'screens', u'seats', u'sheep', u'shelfs', 
      u'shirts', u'shoes', u'shorts', u'sidewalks', u'signs', 
      u'sinks', u'skateboards', u'skies', u'skiers', u'sneakers', 
      u'snows', u'socks', u'stands', u'streets', u'surfboards', 
      u'tables', u'tails', u'ties', u'tiles', u'tires', 
      u'toilets', u'towels', u'towers', u'tracks', u'trains', 
      u'trees', u'trucks', u'trunks', u'umbrellas', u'vases', 
      u'vegetables', u'vehicles', u'waves', u'wheels', u'windows', 
      u'windshields', u'wings', u'wires', u'women', u'zebras']

'''
pl = [u'airplanes', u'animals', u'arm', u'bag', u'bananas',
      u'baskets', u'beaches', u'bears', u'beds', u'benches',
      u'bikes', u'birds', u'boards', u'boats', u'books',
      u'boots', u'bottles', u'bowls', u'boxes', u'boys',
      u'branches', u'buildings', u'buses', u'cabinets', u'caps',
      u'cars', u'cats', u'chairs', u'children', u'clocks',
      u'decks', u'dogs', u'doors', u'drawers', u'ears',
      u'elephants', u'engines', u'eyes', u'faces', u'fences',
      u'fingers', u'flags', u'flowers', u'food', u'forks',
      u'fruits', u'giraffes', u'girls', u'glasses', u'gloves',
      u'guys', u'hairs', u'hands', u'handles', u'hats',
      u'heads' u'helmets', u'hills', u'horses', u'houses', 
      u'jackets', u'jeans', u'kids', u'kites', u'ladies', 
      u'lamps', u'laptops', u'leafs', u'legs', u'letters', 
      u'lights', u'logos', u'men', u'men', u'motorcycles', 
      u'mountains', u'mouths', u'necks', u'noses', u'numbers', 
      u'oranges', u'pants', u'paper', u'paws', u'people', 
      u'persons', u'phones', u'pillows', u'pizzas', u'planes', 
      u'plants', u'plates', u'players', u'poles', u'posts', 
      u'pots', u'rackets', u'railings', u'rocks', u'roofs', 
      u'rooms', u'screens', u'seats', u'sheep', u'shelfs', 
      u'shirts', u'shoes', u'shorts', u'sidewalks', u'signs', 
      u'sinks', u'skateboards', u'skies', u'skiers', u'sneakers', 
      u'snow', u'socks', u'stands', u'streets', u'surfboards', 
      u'tables', u'tails', u'ties', u'tiles', u'tires', 
      u'toilets', u'towels', u'towers', u'tracks', u'trains', 
      u'trees', u'trucks', u'trunks', u'umbrellas', u'vases', 
      u'vegetables', u'vehicles', u'waves', u'wheels', u'windows', 
      u'windshields', u'wings', u'wires', u'women', u'zebras']
'''
pd = [u'above', u'across', u'against', u'along', u'and', 
      u'at', u'attached to', u'behind', u'belonging to', u'between', 
      u'carrying', u'covered in', u'covering', u'eating', u'flying in', 
      u'for', u'from', u'growing on', u'hanging from', u'has', 
      u'holding', u'in', u'in front of', u'laying on', u'looking at', 
      u'lying on', u'made of', u'mounted on', u'near', u'of', 
      u'on', u'on back of', u'over', u'painted on', u'parked on', 
      u'part of', u'playing', u'riding', u'says', u'sitting on', 
      u'standing on', u'to', u'under', u'using', u'walking in', 
      u'walking on', u'watching', u'wearing', u'wears', u'with']

op = [u'below', u'across', u'against', u'beside', u'and',
      u'where xxx at', u'attaching with', u'in front of', u'has', u'surrounding',
      u'being carried by', u'covering', u'covered in', u'being eating', u'in which xxx is flying',
      u'for', u'where xxx is from', u'where xxx grow', u'hanging', u'of',
      u'being held by', u'with xxx in', u'behind', u'with xxx laying on', u'being look by',
      u'with xxx lying on', u'contructing', u'mounted', u'near', u'has',
      u'with xxx on it', u'in front of', u'below', u'being painted by', 'being parked with',
      u'has', u'being played by', u'being ride by', u'being said by', 'being sitted by',
      u'being walked by', u'being watched by', u'be wearing on', u'be worn on', u'with']
      

sy = {'boy': ['kid', 'man', 'person', 'child'],
      'man': ['person', 'boy', 'guy'],
      'woman': ['girl']}

def get_pl(word):
    if word in pl:
        return word
    if not word in sl:
        return None
    return pl[sl.index(word)]

def is_sl(word):
    return word in sl

def is_pl(word):
    return word in pl

def get_op(rel):
    return op[pd.index(rel)]

def get_gc(word):
    if word in pl: return ''
    if word[0] in ['a', 'e', 'i', 'o', 'u']: return 'an '
    else: return 'a '

def get_sy(word):
    if not word in sy.keys(): return []
    else: return sy[word]