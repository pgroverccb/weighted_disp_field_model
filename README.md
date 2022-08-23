## Displacement Field Model for Tracking Nuclei

### Running Saved Model on External Raw Images

The model is part of a bigger end-to-end system for segmentation and tracking, and thus requires instance segmentation masks as input, where background is labelled as 0, and foreground instances are labelled [1, 2, 3 ... ]. For demonstration, U3D-BCD network can be used for this purpose. Additional MONAI library is utilized for sliding window approach (available [here](https://drive.google.com/drive/u/0/folders/1A_q8lcUjO-rUbi0iwppIXCzHofhdZOFX)). Saved model is placed in _utils/u3d_bcd_saved_model.pth_

* **pred_inst_u3d_bcd.py** -  _generates instance segmentation masks from a raw unannotated image._ 

* **lineage_pred_external_disp_field_sliding.py** - _predicts edges between instances present in pair of frames, with the help of predicted instance segmentation masks._

---

### To construct GT dataset :

For each pair of frames, it is possible to construct a displacement field map, where a 3D vector to the centroid of corresponding nuclei is computed for each voxel. In case of division of nuclei, parent nuclei is bisected volumetrically based on location of daughter nuclei, and halves are directed to respective daughter nuclei.

* **compute_disp_field.py** -  _contains function which takes in a pair of frames and computes x, y and z displacement maps._

* **form_dataset.py** - _formulates dataset which has an input of two raw images, and output of 5 channels, two channels for prediction of continuous cell-cycle stage in the pair of frames, and x, y and z displacement maps._

### To train and test network : 

3D U-Net is used for the application, its architecture is adopted from PyTorch Connectomics Library. Displacement vectors for dividing nuclei are much larger than those of non-dividing, forming a weight imbalance as cases of division are quite infrequent. To address this issue, the network divides displacement field volumes into dividing and non-dividing groups based on thresholding from stage predictions. This allows faster and more accurate predictions. 

Supporting data including PyTorch Connectomics Framework (relevant files) and Sample Dataset Files are shared with [Google Drive](https://drive.google.com/drive/u/0/folders/1jBm7_1DwP4E9U5cdcKwdxJOmTYhUifXs)

* **run_disp_field.py** -  _is the script used for training and validating the network, randomized crop to 128 x 128 x 128 is used for each volume._

* **pred_disp_field_sliding.py** - _uses a sliding window apporach predict displacement fields for arbitrary sized volumes, coupled with segmentation masks, choosing an IoU of 0.5, mapping is generated between pair of frames._

* **lineage_pred_disp_field_sliding.py** - _predicts edges between instances present in pair of frames, and measures per-pixel accuracy of mapping._
