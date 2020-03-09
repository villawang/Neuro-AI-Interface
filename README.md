# Neuro-AI-Interface
Use deep neural networks to synthesize the Neuroscore for evaluating Generative Adversarial Networks. Please refer details in our paper ["Synthetic-Neuroscore: Using a neuro-AI interface for evaluating generative adversarial networks"](https://arxiv.org/pdf/1905.04243.pdf) and ["Use of Neural Signals to Evaluate the Quality of Generative Adversarial Network Performance in Facial Image Generation"](https://link.springer.com/article/10.1007/s12559-019-09670-y)

If you find this useful in your research, please consider citing:
```
  @article{wang2019neuroscore,
    title={Synthetic-Neuroscore: Using a neuro-AI interface for evaluating generative adversarial networks},
    author={Wang, Zhengwei and She, Qi and Smeaton, Alan F and Ward, Tomas E and Healy, Graham},
    journal={arXiv preprint arXiv:1905.04243},
    year={2019}
  }
  
  @article{wang2018use,
  title={Use of Neural Signals to Evaluate the Quality of Generative Adversarial Network Performance in Facial Image Generation},
  author={Wang, Zhengwei and Healy, Graham and Smeaton, Alan F and Ward, Tomas E},
  DOI = {https://doi.org/10.1007/s12559-019-09670-y},
  journal={Cognitive Computation},
  year={2019},
  publisher={Springer}
  }
```

## Requirements
This is tensorflow version. A pytorch version might be released in the future if there is lots of demands for that. \
python3\
tensorflow 1.12.0 \
tqdm \
numpy \
sklearn \
pillow \
scipy


## Beamformed EEG data
The beamformed EEG data has been stored in the dropbox link: https://www.dropbox.com/sh/gn10k3qh2i1iq9a/AAD524mfagKemhW0fWK4m_doa?dl=0

GAN images used for training:
https://www.dropbox.com/sh/zw8i4f0o3m0242f/AAA94GlyFyy7ytjdiFvISX_ia?dl=0

Pre-trained model for initialization (Inception v3 and Mobilenet V2 used in this work):
https://www.dropbox.com/sh/8hhcgbmdqa2206j/AAC9L1Cpbzggk6te7F0tYMbaa?dl=0

Model trained by using EEG and without EEG
https://www.dropbox.com/sh/3q7hapmklp5rxvv/AAC6A-Hjt5kp8_PAt6Jo9ipsa?dl=0




