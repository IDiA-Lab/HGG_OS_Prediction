
# Sexually Dimorphic Computational Histopathological Signatures Prognostic of Overall Survival in High-Grade Gliomas via Deep Learning 
Ruchika Verma, Tyler J. Alban, Prerana Parthasarathy, Mojgan Mokhtari, Paula Toro Castano, Mark L. Cohen, Justin D. Lathia, Manmeet Ahluwalia, Pallavi Tiwari
Radiology: Science Advances (Provisionally Accepted on June, 2024)

# Abstract
High-grade glioma (HGG) is an aggressive brain tumor. Sex is an important factor that differentially impacts survival outcomes in HGG. We employed an end-to-end deep-learning approach on Hematoxylin and eosin (H&E) scans to (1) identify explainable, sex-specific histopathological attributes of the tumor microenvironment (TME) that may be associated with patient-outcomes, and (2) create sex-specific risk profiles to prognosticate overall survival. Surgically resected H&E-stained tissue slides were analyzed in a two-stage approach using ResNet18 deep-learning models, first, to segment the viable tumor regions, and second, to build sex-specific prognostic models for prediction of overall survival. Our mResNet-Cox model yielded C-index (0.696, 0.736, 0.731, 0.729) for the female cohort and C-index (0.729, 0.738, 0.724, 0.696) for the male cohort across training and three independent validation cohorts, respectively. End-to-end deep-learning approaches using routine H&E-stained slides, trained separately on male and female HGG patients, may allow for identifying sex-specific histopathological attributes of the TME associated with survival and, ultimately, build patient-centric prognostic risk-assessment models. 

# Main Files:
[ResNet-train](https://github.com/IDiA-Lab/HGG_OS_Prediction/blob/main/ResNet_train.py): Code to train Resnet18 model for Tumor Segmentation
[Tumor_Segmentation](https://github.com/IDiA-Lab/HGG_OS_Prediction/blob/main/Tumor_Segmentation.py): Code to segment tumor regions from WSI using ResNet18 model
[ResNet_Cox_train](https://github.com/IDiA-Lab/HGG_OS_Prediction/blob/main/ResNet_Cox_train.py): Code to train ResNet-Cox model in 5 folds cross-validation setting
[Evaluate_ResNetCox](https://github.com/IDiA-Lab/HGG_OS_Prediction/blob/main/Evaluate_ResNetCox.py): Code to evaluate ResNet-Cox model
