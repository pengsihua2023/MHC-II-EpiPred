# Model description
**MHC-II-EpiPred** (MHC-II-EpiPred, T cell MHC II molecular epitope prediction) is a protein language model fine-tuned from [**ESM2**](https://github.com/facebookresearch/esm) pretrained model [(***facebook/esm2_t33_650M_UR50D***)](https://huggingface.co/facebook/esm2_t33_650M_UR50D).    

**MHC-II-EpiPred** is a classification model that uses potential epitope peptides as input to predict T cell epitopes of MHC-II. The model is fed with a peptide sequence, and the output of the model is whether the peptide is a T cell epitope of MHC-II.  

# Dataset
The original data was downloaded from IEDB data base at https://www.iedb.org/home_v3.php.  
The full data can be downloaded at  https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip  
This dataset comprises 543,717 T-cell epitope entries, spanning a variety of species and infections caused by diverse viruses. The epitope information included encompasses a broad range of potential sources, including data relevant to disease immunotherapy.  

Finally, the dataset we used to train the model contains 60,256 positive and negative samples, which is stored in https://github.com/pengsihua2023/MHC-II-EpiPred/tree/main/data.   

# Results
**MHC-II-EpiPred** achieved the following results:  
Training Loss (cross-entropy loss, CEL): 0.1407   
Training Accuracy: 98.98%  
Evaluation Loss (cross-entropy loss, CEL): 0.0836    
Evaluation Accuracy: 97.03%   
Avg. F1 Score: 98.97%  
Epochs: 324  
Train runtime：20.35 Hours  
GPUs used: 4 H100 with 80G Memory  

![image](https://github.com/user-attachments/assets/bee6d75c-919b-4de3-97f4-0b372e1dd898)  
Figure 2 Training and Evaluation Loss during the training process of the MHC-II-EpiPred model

![image](https://github.com/user-attachments/assets/3df0a0f1-9cdd-431d-9055-127fe0efc3d3)  
Figure 3 Evaluation accuracy during the training process  

# Model at Hugging Face
https://huggingface.co/sihuapeng/MHC-II-TCEpiPred 
# Model deployed
http://72.167.44.178:8000/   
The best performance of the fine-tuned model is the 650M parameter model, but such a model cannot be deployed on the 4G memory server I rented. In other words, the hardware for model reasoning does not meet the minimum requirements. So the model deployed this time is a model with only 8M parameters. This small parameter model is deployed for demonstration purposes.  

# How to use **MHC-II-EpiPred**
### An example
Pytorch and transformers libraries should be installed in your system.  
### Install pytorch
```
pip install torch torchvision torchaudio

```
### Install transformers
```
pip install transformers

```
### Run the following code
```
Coming soon!

```


## Funding
This project was funded by the CDC to Justin Bahl (BAA 75D301-21-R-71738).  
### Model architecture, coding and implementation
Sihua Peng  
## Group, Department and Institution  
### Lab: [Justin Bahl](https://bahl-lab.github.io/)  
### Department: [College of Veterinary Medicine Department of Infectious Diseases](https://vet.uga.edu/education/academic-departments/infectious-diseases/)  
### Institution: [The University of Georgia](https://www.uga.edu/)  

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64c56e2d2d07296c7e35994f/2rlokZM1FBTxibqrM8ERs.png)
