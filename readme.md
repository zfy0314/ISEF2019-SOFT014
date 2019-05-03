# ISEF 2019 SOFT014 Enhanced Image Caption Using Scene-Grapg Generation
## notes: 
* This code is for demo only, training and testing will be released later (hopefully)
* A special thank to China Association for Science and Technology for the oppotunity to participate in ISEF
## Pre-requisite
1. NVIDIA GPU with proper drive (384 or later), CUDA (8.0, 9.0, 9.1, 9.2), CUDNN, and NCCL installed
2. install pytorch 0.4.0 or 0.4.1 compatible with CUDA library (pytorch 1.0 is not supported without proper modification) according to [pytorch.org](pytorch.org)
3. (optional for scene-graph visualization) install graphviz: 
```
sudo apt-get install graphviz
```
4. (recommanded) create a new conda virtual environment:
```
conda create -n caption python=3
```
5. clone:
```
git clone https://github.com/zfy0314/ISEF2019-SOFT014.git
```
6. install python packages:
```
pip install -r requirements.txt # add --user if necessary
```
7. setup
```
cd lib
sh make.sh
```
## Download Pretrained Models
1. download Gogle word2vec vocabulary from [here](https://code.google.com/archive/p/word2vec/)
2. download pretrained caption encoder model and decode model from [here](https://drive.google.com/open?id=1039_0YaMubt6J1IBgYm_bROmeXe0RfRX) and [here](https://drive.google.com/open?id=11yfYNqbcAYCm7OvgzPkQYVflVrTyAct9)
3.  download pretrained scene-graph generation models from [here](https://drive.google.com/open?id=1TVeSq1ggoSmi6bbrjumjgqx8TQudO9KQ) and [here](https://drive.google.com/open?id=1TbUOZCNywCeHmk5EqsTRSPzafHf3cOoM)
4. place all the downloaded files into ```pretained/```
## Directory Structure
the final directory should be like this:
```
|--configs/
|  |--e2e_relcnn_X-101-64x4d-FPN_8_epochs_vg_y_loss_only.yaml
|--lib/
|  |--caption/
|  |--core/
|  |--...
|  |--make.sh
|  |--setup.py
|--pretrained/
|  |--decoder-16-4000.ckpt
|  |--det_model_step119999.pth
|  |--encoder-16-4000.ckpt
|  |--GoogleNews-vectors-negative300.bin
|  |--rel_model_step125445
|--tools/
|  |--_init_paths.py
|  |--demo.py
|--readme.md
|--requirements.txt
```
# Run Demo
```
python tools/demo.py --image PATH_TO_IMAGE
```
# Sample Result
