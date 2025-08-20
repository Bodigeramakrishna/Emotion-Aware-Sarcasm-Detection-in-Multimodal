# Emotion-Aware-Sarcasm-Detection-in-Multimodal
**Author**: Ramakrishna Bodige  
**Email**: ramakrishnamail4u@gmail.com  
**Institution**: Department of Computer Science, SR University, Warangal, Telangana, India
# 1. Project Statement
HCI-EASD (Hierarchical Cross-Modal Incongruity and Emotion-Aware Sarcasm Detection) The model integrates four key modules: <br>1.multimodal feature extraction using BERT and ResNet-50<br> 2.emotion-aware processing with incongruity correlation analysis.<br>3.cross-modal incongruity detection through multi-head attention.<br>4.hierarchical classification with compound loss functions.<br>Our approach validates the hypothesis that sarcasm emerges from quantifiable cross-modal discord, enabling detection of both explicit semantic patterns and implicit contradictions.Extensive evaluation on MEMOTION and MUStARD datasets demonstrates exceptional performance with accuracy of 0.97, AUC of 0.991, calibration error of 0.0994, and F1-scores exceeding 0.95 for non-sarcastic content while maintaining robust sarcastic classification above 0.6.This HCI-EASD as an effective framework for multimodal sarcasm detection.

# Contributions:
•	Novel Cross-Modal Incongruity Module explicitly quantifies deliberate text-image mismatches indicating sarcasm.<br>
•	Emotion-Aware Processing Module analyzes emotional incongruity patterns beyond traditional sentiment analysis approaches.<br>
•	Hierarchical architecture with compound loss function teaching underlying sarcasm mechanisms through regularization.<br>
•	Superior interpretable framework achieving 0.991 AUC with comprehensive token-level and feature visualization.<br>

# 2. Methodlogy
**Text to Text Transformer Model**:<br> 
The implemented model is based on the T5 architecture, which operates in a unified framework by converting all NLP tasks into a text-to-text format. This specific implementation consists of three primary components: the embedding layer, the encoder stack, and the decoder stack. Each component plays a pivotal role in the overall functioning of the model.
# 3.	Data Set
The data set used for training the proposed model is collected from Twitter data set, and reddit data set. Totally the numbers of samples are 79813, distributed over 4 classes.  And it is observed that the class-wise distribution of the dataset used to train the proposed model.<br> 
**Irony:** Representing the largest class, it contains 20,894 samples. These entries are primarily focused on capturing ironic statements, which are often characterized by a mismatch between literal and intended meaning.<br>
**Sarcasm:** This class has 20,681 samples and targets sarcastic expressions, which include statements intended to mock or convey contempt, often using humor or exaggeration.<br>
**Humor:** Comprising 19,643 samples, this class focuses on entries that are inherently humorous or light-hearted, capturing nuances that distinguish them from irony and sarcasm.<br>
**Regular:** With 18,595 samples, this class includes statements that do not exhibit any of the characteristics of irony, sarcasm, or humor, serving as a baseline for normal textual data.<br>
# 4.	Result Analysis
The proposed model is trained for with a learning rate of 5*10^(-5) to ensure steady optimization, and a batch size of 16 for both training and evaluation to maintain computational efficiency. The model was trained for 5 epochs, with weight decay set to 0.01 to prevent over fitting. Training is conducted over 5 epochs, leveraging mixed-precision (fp16) to accelerate computations on GPUs. Logs are generated every 10 steps, and logging is directed to a designated directory for monitoring the training progress<br>
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
