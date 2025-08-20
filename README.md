# Emotion-Aware-Sarcasm-Detection-in-Multimodal
**Author**: Ramakrishna Bodige  
**Email**: ramakrishnamail4u@gmail.com  
**Institution**: Department of Computer Science, SR University, Warangal, Telangana, India
# 1. Project Statement
HCI-EASD (Hierarchical Cross-Modal Incongruity and Emotion-Aware Sarcasm Detection) This model integrates four key modules: <br>1.Multimodal feature extraction using BERT and ResNet-50<br> 2.Emotion-aware processing with incongruity correlation analysis.<br>3.Cross-modal incongruity detection through multi-head attention.<br>4.Hierarchical classification with compound loss functions.<br>Our approach validates the hypothesis that sarcasm emerges from quantifiable cross-modal discord, enabling detection of both explicit semantic patterns and implicit contradictions.Extensive evaluation on MEMOTION and MUStARD datasets demonstrates exceptional performance with accuracy of 0.97, AUC of 0.991, calibration error of 0.0994, and F1-scores exceeding 0.95 for non-sarcastic content while maintaining robust sarcastic classification above 0.6.This HCI-EASD as an effective framework for multimodal sarcasm detection.

# Contributions:
•	Novel Cross-Modal Incongruity Module explicitly quantifies deliberate text-image mismatches indicating sarcasm.<br>
•	Emotion-Aware Processing Module analyzes emotional incongruity patterns beyond traditional sentiment analysis approaches.<br>
•	Hierarchical architecture with compound loss function teaching underlying sarcasm mechanisms through regularization.<br>
•	Superior interpretable framework achieving 0.991 AUC with comprehensive token-level and feature visualization.<br>

# 2. Methodlogy
**HCI-EASD model**:<br> 	
Sarcasm detection represents a complex natural language understanding task that requires sophisticated modeling of linguistic nuances, contextual incongruities, and multimodal information. Given a multimodal input consisting of text T and image I, our objective is to learn a mapping function f:(T,I)→{0,1} that accurately classifies sarcastic content. The proposed HCI-EASD model addresses this challenge through a novel hierarchical architecture that explicitly models cross-modal incongruities and integrates emotion-aware features.
# 3. Dataset Analysis and Preprocessing
The experimental evaluation employs two prominent multimodal sarcasm detection datasets: MEMOTION and MUStARD, which collectively provide a comprehensive foundation for training and evaluating the HCI-EASD model.<br> 
**MEMOTION Dataset:** The MEMOTION dataset contributes 5,000 samples to our training corpus, representing a substantial collection of meme-based sarcastic content.This dataset is particularly valuable because it captures the internet meme culture where sarcasm often manifests through deliberate incongruities between visual content and textual overlays.<br>
**MUStARD Dataset:** The MUStARD dataset provides 3,000 samples with a more balanced distribution, containing 62.4% non-sarcastic and 37.6% sarcastic instances. This dataset originates from television show dialogues and captures a different form of sarcastic expression.<br>
**Dataset Overview Analyses**
<img width="1016" height="704" alt="image" src="https://github.com/user-attachments/assets/9d902507-1faa-4ee7-ad00-1014a5fd6156" />
**Sarcasm Pattern Analyses**
<img width="974" height="354" alt="image" src="https://github.com/user-attachments/assets/9c690f21-b571-48b0-8afc-0b847cf2de5e" />
**Training Methodology**
The HCI-EASD model was implemented using the PyTorch deep learning framework, leveraging its dynamic computational graph capabilities for efficient multimodal processing. The model was trained on Kaggle's Tesla T4 GPU with 16GB memory, enabling efficient processing of multimodal batches while maintaining computational accessibility for reproducible research. The training environment utilized CUDA acceleration with automatic mixed precision training to optimize memory utilization and training speed. Batch sizes were dynamically adjusted based on sequence lengths and image resolutions to maximize GPU memory efficiency while ensuring stable gradient computation.


# 4. Experimental Results and Analysis
The HCI-EASD model demonstrates exceptional performance across all evaluation metrics, achieving robust classification accuracy that validates our architectural design Over all accuracy with 0.97 and F1 score as 0.97 illustrates model performing better in all directions.<br>
<img width="742" height="571" alt="image" src="https://github.com/user-attachments/assets/73e31c57-7aae-4511-bedb-2286b0c2b6ba" />

<img width="788" height="611" alt="image" src="https://github.com/user-attachments/assets/300db5ae-750c-421f-95c9-099d519f6aac" />
<img width="736" height="611" alt="image" src="https://github.com/user-attachments/assets/e46f7daa-4e1d-4741-a77e-0fa0bab81f5e" />
<img width="975" height="461" alt="image" src="https://github.com/user-attachments/assets/d46c387f-d7ea-4e19-a9e9-004df1ee127f" />





**Table 1**: Sarcasm detection comparison of proposed model performance with existing models

| System                  | Model                                      | Data set                         | Accuracy | P      | R      | F1     |
|-------------------------|--------------------------------------------|----------------------------------|----------|--------|--------|--------|
| Wang et al. (2020)      | TextCNN with Lexicon Features              | Reddit Sarcasm                   | 88.7     | 87.2   | 88.1   | 87.7   |
| Guo et al. (2021)       | Multi-View Learning with LSTM              | Weibo                            | 90.2     | 89.5   | 90.0   | 89.7   |
| Liu et al. (2022)       | Bi-LSTM                                    | Twitter                          | 73       | 71.8   | 71.7   | 71.7   |
| Yuan et al. (2022)      | Graph Convolutional Network (GCN)          | Social Media Data                | 87.9     | 86.7   | 87.3   | 87.0   |
| Shaheen and Nigam (2022)| Sentiment-Aware Sarcasm Detection          | SemVal-2018                      | 82       | 82     | 77     | 79     |
| Shekhawat et al. (2022) | Hybrid Model with SVM and CNN              | User-Generated Content           | 85.7     | 84.3   | 85.1   | 84.7   |
| Liu et al. (2023)       | RoBERT                                     | IAC-V1, IAC-v2, twitter*         | 77.1     | 78.3   | 76.6   | 76.9   |
| Li et al. (2021)        | Attention Mechanism with Bi-LSTM           | Multi-Platform Sarcasm Dataset   | 88.0     | 86.5   | 87.0   | 86.7   |
| Eke et al. (2021)       | BERT with User Profiling                   | Twitter                          | 89.2     | 88.0   | 88.8   | 88.4   |
| Potamias et al. (2020)  | Deep Learning with Lexical Features        | Reddit, SemVal-2018              | 85,89    | 78,81  | 78,80  | 78,80  |
| Tomás et al. (2023)     | Generative Adversarial Networks (GANs)     | Mixed Media Dataset              | 90.5     | 89.8   | 90.0   | 89.9   |
| **T5 (Text-to-Text Transfer Transformer) model**      | **Text to Text transfer Transformer model**| **Twitter(multi class) sarcasm** | **96**   | **93** | **88** | **91** |
## 5. Conclusion
This paper introduces a novel T5-based approach for detecting sarcasm, irony, and humor in social media text. Leveraging T5's adaptability for multi-class sarcasm classification, our model effectively captures complex linguistic features in these expressions. Experiments demonstrate state-of-the-art performance with 96% accuracy.

Results show T5's superiority in handling challenging aspects like cultural context, overstatement, and idioms - areas where traditional BERT models struggle. Our balanced, diverse dataset enables robust generalization across sarcasm types.

This work advances sarcasm detection and establishes a foundation for applying transfer learning to more complex humor and irony tasks.
