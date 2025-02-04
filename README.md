# Model description
**MHC-II-EpiPred** (MHC-II-EpiPred, T cell MHC II molecular epitope prediction) is a protein language model fine-tuned from [**ESM2**](https://github.com/facebookresearch/esm) pretrained model [(***facebook/esm2_t33_650M_UR50D***)](https://huggingface.co/facebook/esm2_t33_650M_UR50D).    

**MHC-II-EpiPred** is a classification model that uses potential epitope peptides as input to predict T cell epitopes of MHC-II. The model is fed with a peptide sequence, and the output of the model is whether the peptide is a T cell epitope of MHC-II.  

**MHC-II-EpiPred** achieved the following results:  
Training Loss (mse): 0.1407   
Training Accuracy: 0.9898  
Evaluation Loss (mse): 0.0836    
Evaluation Accuracy: 0.9703    
Epochs: 324  
# The dataset for training **MHC-II-EpiPred**
The original data we obtained comes from the data in the paper by [Lee CH et al.](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-023-01225-z) The data is in a CSV file with a total of 9 columns with a sample size of 100,097. We used the first column (amino acid sequences), the second column (immunogenicity, positive or negative), and the ninth column (immunogenicity score). We used these three columns as input to fine-tune the ESM2 pre-trained model and built a regression model. Using this regression model, by inputting potential epitope amino acid sequences, we can predict the immunogenicity score of the potential epitope, and then determine whether it is an epitope based on the set threshold.

The dataset was downloaded from GtHub at [**TRAP**](https://github.com/ChloeHJ/TRAP/blob/main/data/pathogenic_db.csv). 

# Model at Hugging Face
[https://github.com/pengsihua2023/MHC-II-EpiPred](https://huggingface.co/sihuapeng/MHC-II-EpiPred)   

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_name = "sihuapeng/PPPSL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sample protein sequence
protein_sequence = "MSKKVLITGGAGYIGSVLTPILLEKGYEVCVIDNLMFDQISLLSCFHNKNFTFINGDAMDENLIRQEVAKADIIIPLAALVGAPLCKRNPKLAKMINYEAVKMISDFASPSQIFIYPNTNSGYGIGEKDAMCTEESPLRPISEYGIDKVHAEQYLLDKGNCVTFRLATVFGISPRMRLDLLVNDFTYRAYRDKFIVLFEEHFRRNYIHVRDVVKGFIHGIENYDKMKGQAYNMGLSSANLTKRQLAETIKKYIPDFYIHSANIGEDPDKRDYLVSNTKLEATGWKPDNTLEDGIKELLRAFKMMKVNRFANFN"

# Encode the sequence as model input
inputs = tokenizer(protein_sequence, return_tensors="pt")

# Perform inference using the model
with torch.no_grad():
    outputs = model(**inputs)

# Get the prediction result
logits = outputs.logits
predicted_class_id = torch.argmax(logits, dim=-1).item()
id2label = {0: 'CYtoplasmicMembrane', 1: 'Cellwall', 2: 'Cytoplasmic', 3: 'Extracellular', 4: 'OuterMembrane', 5: 'Periplasmic'}
predicted_label = id2label[predicted_class_id]

# Output the predicted class
print ("===========================================================================================================================================")
print(f"Predicted class Label: {predicted_label}")
print ("===========================================================================================================================================")

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
