Source code for "Automated Fusion of Multimodal Electronic Health Records for Better Medical Predictions"
(https://arxiv.org/abs/2401.11252)

1. Access the MIMIC - III dataset (https://physionet.org/content/mimiciii/1.4/)
1. Following Fiddle Pipeline to process the MIMIC-III data (https://github.com/MLD3/FIDDLE-experiments).
2. Note that we seperatly process the discrete events and continues events as two time series modalities, which can be achieved through minimal changes to the original Fiddle pipeline. As show in the figure below, two groups of tables are extracted in the original pipeline. Instead of combining them together, we extract one group at a time and run the pipeline for two times to obtain the corresponding time series. 
![image](https://github.com/SH-Src/AUTOMF/assets/51844791/75a6333e-562f-4462-85e3-102fdbf94304)
3. Put the extracted features and mimic tables at specified locations.
![image](https://github.com/SH-Src/AUTOMF/assets/51844791/8ef17950-a3ca-4bf7-a806-d5e04f5333fc)
4. Run the main scripts.
```
python search_auc.py
```

Please consider cite our paper:
```
@article{cui2024automated,
  title={Automated fusion of multimodal electronic health records for better medical predictions},
  author={Cui, Suhan and Wang, Jiaqi and Zhong, Yuan and Liu, Han and Wang, Ting and Ma, Fenglong},
  journal={arXiv preprint arXiv:2401.11252},
  year={2024}
}
```
