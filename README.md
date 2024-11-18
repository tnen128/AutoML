# AutoML with Fine-Tuned LLMs for Multivariate Regression

## Project Overview

This project explores the potential of fine-tuning large language models (LLMs) to automate the selection of machine learning algorithms for multivariate regression tasks. Given a knowledge base of meta-features, dataset descriptions, and corresponding recommended algorithms, we fine-tune the following LLMs using LoRA (Low-Rank Adaptation):  

1. **Llama 3.2 (3B)** or [piotr25691/thea-3b-25r](https://huggingface.co/piotr25691/thea-3b-25r)  
2. **gemma2:2b**  
3. **phi3.5: 3.8B**  

Each model is fine-tuned and evaluated for its ability to recommend the most suitable algorithm based on dataset meta-features. The fine-tuning process involves adapting the models to generate accurate algorithm recommendations given a dataset description.

## Repository Structure

The repository contains three Jupyter notebooks, one for each model:  
- **`Llama3.2_finetuning.ipynb`**  
- **`gemma2_finetuning.ipynb`**  
- **`phi3.5_finetuning.ipynb`**

These notebooks include:  
- Preprocessing of meta-features and descriptions.  
- LoRA fine-tuning steps.  
- Model evaluation.

## Models and Tokenizers

The models are included in this link [[Models](https://drive.google.com/drive/folders/1zGTdDyM4r7b5VIINH_dKVpS0Vp1e4gT_?usp=sharing)]
