# Medical Report Summarization using Medical Knowledge - Planning 

## Implementation Process

### Step 1: Data Collection

1. **Collect Datasets**  
   - Use datasets like **IU-Xray** and **MIMIC-CXR** for training and testing.


### Step 2: Data Preprocessing

1. **Preprocess Images**  
   - Resize images to the required dimensions (e.g., `224x224`).  
   - Normalize pixel values based on the dataset's statistics.

2. **Preprocess Text**  
   - Tokenize medical reports and convert them into sequences.  
   - Create a vocabulary mapping words to integer indices.  
   - Pad sequences to a uniform length for consistency.

3. **Create Data Loaders**  
   - Implement data loaders to manage batching for both images and text.


### Step 3: Model Implementation

1. **Visual Extractor**  
   - Implement the `VisualExtractor` class using a pre-trained **ResNet** model to extract visual features from X-ray images.

2. **Text Encoder**  
   - Implement the `TextEncoder` class with an embedding layer and **LSTM** to encode textual input from the medical reports.

3. **Multilevel Alignment**  
   - Implement the `MultilevelAlignment` class to align and combine visual and text features at multiple levels.

4. **Report Generator**  
   - Implement the `ReportGenerator` class using an **LSTM** decoder to generate medical reports from the aligned features.

5. **Complete Model**  
   - Implement the `MedicalReportGenerationModel` class that integrates all components (Visual Extractor, Text Encoder, Multilevel Alignment, and Report Generator) into one model for report generation.


### Step 4: Training

1. **Training**  
   - Train the complete model using the prepared datasets.


### Step 5: Testing  

1. **Testing**  
   - Test the model on unseen X-ray images and evaluate its performance.


