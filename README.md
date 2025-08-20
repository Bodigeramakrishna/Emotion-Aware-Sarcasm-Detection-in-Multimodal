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

**Dataset Overview Analyses**:
<img width="1016" height="704" alt="image" src="https://github.com/user-attachments/assets/9d902507-1faa-4ee7-ad00-1014a5fd6156" />

**Sarcasm Pattern Analyses**:
<img width="974" height="354" alt="image" src="https://github.com/user-attachments/assets/9c690f21-b571-48b0-8afc-0b847cf2de5e" />
**Training Methodology**:<br>
The HCI-EASD model was implemented using the PyTorch deep learning framework, leveraging its dynamic computational graph capabilities for efficient multimodal processing. The model was trained on Kaggle's Tesla T4 GPU with 16GB memory, enabling efficient processing of multimodal batches while maintaining computational accessibility for reproducible research. The training environment utilized CUDA acceleration with automatic mixed precision training to optimize memory utilization and training speed. Batch sizes were dynamically adjusted based on sequence lengths and image resolutions to maximize GPU memory efficiency while ensuring stable gradient computation.
# 4. Experimental Results and Analysis
The HCI-EASD model demonstrates exceptional performance across all evaluation metrics, achieving robust classification accuracy that validates our architectural design Over all accuracy with 0.97 and F1 score as 0.97 illustrates model performing better in all directions.<br>
The Receiver Operating Characteristic (ROC) curve analysis provides compelling evidence of the model's discriminative power, with an Area under the Curve (AUC) of 0.991. This exceptional AUC score indicates that the model can distinguish between sarcastic and non-sarcastic content with remarkable accuracy across all decision thresholds.<br>
**Classification Performance Metrics and ROC Analysis**<br>

<img width="975" height="372" alt="image" src="https://github.com/user-attachments/assets/79cfb4c8-2344-4a20-b767-c02031776195" />

**Cross-Modal Incongruity Analysis and Validation**:<br>
The incongruity score analysis provides critical validation of core hypothesis that sarcastic content exhibits measurable cross-modal discord.the Cross-Modal Incongruity Module successfully captures the deliberate mismatches characteristic of sarcastic communication.\<br>

**Incongruity Score Analysis and Prediction Accuracy Distribution**<br>
<img width="975" height="314" alt="image" src="https://github.com/user-attachments/assets/e6a7da00-ad91-463b-8170-6acb6f527a86" />
**Model Confidence Distribution for Sarcastic and Non-Sarcastic Classifications**<br>
<img width="975" height="484" alt="image" src="https://github.com/user-attachments/assets/1b46e6cc-73e2-45e9-9a8b-8f76b1dfc00a" />
**Model Calibration assessment and Reliability Analysis**<br>
<img width="975" height="779" alt="image" src="https://github.com/user-attachments/assets/26913435-17bb-4d41-964e-1052ed2ae8fe" />
**Error Analysis of Challenging Cases and Feature Correlation Matrix**<br>
<img width="975" height="503" alt="image" src="https://github.com/user-attachments/assets/b5158e91-b952-4704-b485-9970ca727462" />
**Feature Attribution Analysis and Importance Distribution**<br>
<img width="975" height="779" alt="image" src="https://github.com/user-attachments/assets/c2238451-6a9c-4e5b-ae32-afb0370e4d4e" />
**Token-Level Importance Analysis for Sarcasm Prediction with Text Highlighting**<br>
<img width="975" height="446" alt="image" src="https://github.com/user-attachments/assets/01fde011-3fdf-472d-841f-fd0ec591fd01" />







 




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
HCI-EASD, a novel hierarchical architecture for multimodal sarcasm detection that successfully addresses the challenge of identifying sarcastic expressions through explicit cross-modal incongruity modeling and emotion-aware processing.The Cross-Modal Incongruity Module effectively captures deliberate mismatches between textual and visual content, while the emotion-aware component enhances detection by analyzing emotional context patterns. Experimental validation on MEMOTION and MUStARD datasets demonstrates exceptional performance with AUC of 0.991 and calibration error of 0.0994, confirming reliable uncertainty quantification. The model achieves balanced performance with F1-scores exceeding 0.95 for non-sarcastic content and robust sarcastic classification above 0.6.The HCI-EASD framework establishes a significant contribution to multimodal learning, providing both superior performance and enhanced interpretability for real-world sarcasm detection applications in social media analysis and content moderation systems.
