# Image Retrieval for Breast Histology Images

This repo compares different ways to do **content-based image retrieval** on breast cancer histology images from the **BRACS** dataset. Given one tissue image, the goal is to find other images of the same class.

Each method turns every image into a vector (an *embedding*). To search, we find the closest vectors (k-nearest neighbours). A search counts as correct when the closest image has the same class as the query.

All work is in **Jupyter notebooks made for Google Colab**.

## Data

- **Dataset:** BRACS image patches, 7 classes (folders `0` to `6`).
- **Only 3 classes are used:** `0`, `3` and `5`. The notebooks delete folders `1`, `2`, `4` and `6` after unzipping.
- **Three zip files** (on Google Drive):

| File | Content | Google Drive file ID |
|---|---|---|
| `train.zip` | Training images (about 6 GB) | `1BRccmUBnxhbthHL0bxVz1nLD7OSbZ3r2` |
| `bracs_in_test.zip` | Test images, same distribution as training ("in-distribution") | `13mgPPoiktXLovSHvV13vmz-1W6F6EwJq` |
| `bracs_ood_test.zip` | Test images from a different distribution ("out-of-distribution", OOD) | `1Y-hVa9N-AyabzIc-KKqr5TTwH-tdrwin` |

- **Patient ID:** the code treats the last number in an image's file name as its patient ID. Some evaluations skip neighbours from the same patient.

## Methods (folders)

| Folder | What it does |
|---|---|
| `Supervised/` | Supervised deep metric learning with the [DML_HistoImgRetrieval](https://github.com/easonyang1996/DML_HistoImgRetrieval) code. Model: SE-ResNet-50, loss: Multi-Similarity loss. |
| `Dino/` | Self-supervised learning with [DINO](https://github.com/facebookresearch/dino) (ViT-Small, patch size 8), starting from DINO's official weights. |
| `MAE/UsingPretrain/` | Continues self-supervised [MAE](https://github.com/facebookresearch/mae) training (ViT-Base) on BRACS, starting from the official `mae_finetuned_vit_base.pth` checkpoint. |
| `MAE/UsintFinetune/` | Fine-tunes the MAE ViT-Base model as a 3-class classifier, then uses its features as embeddings. |
| `SimCLR/` | Self-supervised SimCLR with the [lightly](https://github.com/lightly-ai/lightly) library. Three versions: ResNet-50 from the official SimCLR checkpoint, ResNet-18 (ImageNet weights), and ViT-B/16 (ImageNet weights). |
| `VICReg/` | Self-supervised VICReg with lightly. Three versions: ResNet-50 from the official VICReg checkpoint, ResNet-18 (ImageNet weights), and ViT-B/16 (ImageNet weights). |
| `ModularRetrieval/` | A pretrained backbone plus one small layer that outputs a 256-number embedding, trained with triplet loss. Backbone options in the code: CLIP ViT-B/32, torchvision ResNet-50, timm ResNet-50 (the notebook uses timm ResNet-50). |
| `DifferentLossFunctions/` | Same idea as ModularRetrieval with a timm ResNet-18, to try different losses from `pytorch-metric-learning` (Triplet, Circle, Contrastive, Intra-Pair Variance, Generalized Lifted Structure, Margin). Choose one with `loss_type`. |
| `CustomKNN/` | Evaluates saved embeddings: top-1 retrieval accuracy, with an option to skip neighbours from the same patient. `CustomKNN_Union` counts a search as correct if **DINO or the supervised model** finds the right class. |
| `CrossEncoderWithCLSEmbeddings/` | A re-ranker that looks at **two** images and predicts if they match (same class, different patient). `Create_CSV_File` builds training pairs: for each training image, 5 matching and 5 non-matching neighbours found with DINO and the supervised model. The 3 models (MLP, Transformer encoder, full Transformer) take the DINO embeddings of both images. |
| `CrossEncoderWithPatchEmbeddings/` | Same re-ranker idea, but a Transformer encoder reads the DINO **patch** embeddings of both images (computed from the images during training). |

Most method folders have two notebooks:
- `TrainAndResume.ipynb`: trains the model. It can resume from a checkpoint.
- `GetEmbedding.ipynb`: loads the final checkpoint and saves embeddings, labels and file names.

SimCLR, VICReg and DifferentLossFunctions do everything in one notebook.

## Results (from the saved notebook outputs)

| Method | Metric | Test set | Result |
|---|---|---|---|
| Supervised (SE-ResNet-50, MS loss) | Top-1 retrieval accuracy | OOD | 82.3% |
| DINO (ViT-S/8) | 20-NN classification, top-1 | OOD | 73.2% |
| MAE fine-tuned (ViT-B) | Classification accuracy | In-distribution / OOD | 88.0% / 75.4% |
| ModularRetrieval | Top-1 retrieval, skip same patient (CustomKNN) | OOD | 32.5% |
| DINO + Supervised (CustomKNN_Union) | Top-1 retrieval, skip same patient, either correct | OOD | 76.6% |
| Cross-encoder, MLP | Pair accuracy | Validation pairs | 87.5% |
| Cross-encoder, Transformer versions | Pair accuracy | Validation pairs | about 50% (did not learn) |

The patch-embedding cross-encoder was stopped early (one epoch would take many hours), so it has no result.

## How to run

### What you need

- **Google Colab with a GPU** (*Runtime → Change runtime type → GPU*). The training code expects CUDA.
- **A Google account**, because almost every notebook mounts Google Drive.

### 1. Prepare Google Drive

1. Put the three zip files in a folder named **`bracs`** in your Drive: `MyDrive/bracs/`. The MAE, SimCLR, VICReg and DifferentLossFunctions notebooks copy the data from there. The other notebooks download it themselves with `gdown`.
2. Create these empty folders in `MyDrive/`. The notebooks save checkpoints and results there:
   `ImageRetrievalSupervised`, `DINO`, `MAE`, `ModularRetrieval`, `CrossEncoderEmbeddings`.

### 2. Run a method

1. Open the notebook in Colab (*File → Upload notebook*, or open it from GitHub).
2. Run the cells from top to bottom.
3. Watch for the **colored text cells**. They tell you what to do:
   - **"Change ... file"**: edit that file in the cloned repo (DINO, MAE or DML_HistoImgRetrieval) as described, before running the next cell. Open the file in Colab's file browser (left panel), change it and save.
   - **"Don't run this cell, replace ... with this code"**: copy the cell's code into the named file instead of running it.
4. **Checkpoints:** many notebooks download the trained checkpoints from the original runs with `gdown`. To train from scratch, skip those cells. To resume your own training, put your checkpoint where the notebook says (usually named `checkpoint.pth`).
5. For methods with two notebooks, run `TrainAndResume` first, then `GetEmbedding`.

**Save your outputs.** Files saved in the Colab working folder (like `./embeddings.pth`) are deleted when the session ends. Copy them to Drive or download them.

### 3. Evaluate with CustomKNN

`CustomKNN` does not need a GPU or Drive. It only needs `torch`, `numpy`, `scikit-learn` and `tqdm`. Put the embedding files next to the notebook with these names:

**`CustomKNN.ipynb`** (set `which_code` to `"dino"`, `"supervised"` or `"modular"`):

| `which_code` | Files the notebook expects | Made by |
|---|---|---|
| `dino` | `2-testfeat.pth`, `2-testlabels.pth`, `2-file_names.json` | `Dino/GetEmbedding` (`testfeat.pth`, `testlabels.pth`, `file_names.json`, renamed) |
| `supervised` | `2-embeddings.npy`, `2-labels.npy`, `2-test_file_names.json` | `Supervised/GetEmbedding` (`embeddings.npy`, `labels.npy`, `test_file_names.json`, renamed) |
| `modular` | `embeddings.npy`, `labels.npy`, `file_paths.json` | `ModularRetrieval/GetEmbedding` |

**`CustomKNN_Union.ipynb`** expects `dino_embeddings.pth`, `dino_labels.pth`, `dino_file_names.json` and `supervised_embeddings.npy`, `supervised_labels.npy`, `supervised_file_names.json` (the same files as above, renamed).

Set `consider_same_patients = True` to skip neighbours from the same patient.

### 4. Run the cross-encoders

1. (Optional) Run `CrossEncoderWithCLSEmbeddings/Create_CSV_File.ipynb` to build the pairs file `csv_for_k_5.csv`. It takes about an hour. You can skip this: the cross-encoder notebooks download a ready CSV.
2. Run any cross-encoder notebook. They download everything they need with `gdown`: the CLS-embedding notebooks download saved DINO and supervised embeddings; the patch-embedding notebook downloads the images and the trained DINO checkpoint.

## Known issues

- **`Supervised/TrainAndResume`:** the checkpoint download cell saves to `./gdrive/...`, but `train.py` looks for `../gdrive/MyDrive/ImageRetrievalSupervised/checkpoint.pth`. If you run the cells in order, change the download path to `../gdrive/MyDrive/ImageRetrievalSupervised/checkpoint.pth`. Otherwise training starts from epoch 0.
- **Supervised notebooks:** the note says to change `data_preprocess.py`, but the file is `KIMIA_data/data_process.py` (lines 4 and 5).
- **`DifferentLossFunctions`:**
  - It installs `pytorch-metric-learning` from a fork (`github.com/nazari02105/pytorch-metric-learning`).
  - The checkpoint cell downloads the **ModularRetrieval** checkpoint (ResNet-50). It does not fit this ResNet-18 model and loading it fails, so skip that cell.
  - Resuming from this notebook's own checkpoints also fails, because the code loads `checkpoint['loss']` but never saves it.
  - `loss_type = "MultiSimilarityLoss"` actually creates `MarginLoss`.
- **SimCLR with official checkpoint:** the download link in the notebook expires. Get a new link from the [SimCLR GitHub page](https://github.com/google-research/simclr) each time.
- **`SimCLR_ViT` and all three VICReg notebooks:** the embedding step loads a fixed path like `./lightning_logs/version_0/checkpoints/epoch=1-step=1858.ckpt`. Change it to the checkpoint your training run created (look in the `lightning_logs` folder).

## Troubleshooting

- **"Weights only load failed" when loading a checkpoint:** since PyTorch 2.6, `torch.load` only loads plain weights by default, and DINO and MAE checkpoints contain more than that. Add `weights_only=False` to the `torch.load(...)` call named in the error. Only do this for checkpoints you trust.
- **Errors from `timm`:** the notebooks were run with `timm 0.9.12` (Python 3.10, January 2024). If a newer version breaks, install that one: `pip install timm==0.9.12`.
- **`gdown` says "Too many users have viewed or downloaded this file":** Google Drive limits downloads. Wait and try again later, or copy the file to your own Drive.
