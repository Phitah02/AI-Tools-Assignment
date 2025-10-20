# AI Tools and Applications Assignment: "Mastering the AI Toolkit" 🛠️🧠

## Project Overview

This comprehensive assignment evaluates understanding of AI tools/frameworks and their real-world applications through theoretical, practical, and ethical tasks. It demonstrates proficiency in selecting, implementing, and analyzing AI tools to solve problems across different domains including classical machine learning, deep learning, and natural language processing.

**Theme:** "Mastering the AI Toolkit"  
**Duration:** 7 days  
**Objective:** Showcase AI toolkit mastery through hands-on implementation of three core AI tasks

### Learning Objectives
- Master classical machine learning with Scikit-learn for classification tasks
- Implement deep learning models using TensorFlow/Keras for image recognition
- Apply NLP techniques with spaCy for text analysis and entity recognition
- Understand ethical considerations and bias mitigation in AI systems
- Deploy AI models using modern web frameworks (Streamlit)
- Demonstrate proficiency in data preprocessing, model evaluation, and visualization

## Assignment Structure

### Part 1: Theoretical Understanding (40%)
- **Short Answer Questions:**
  - Q1: Differences between TensorFlow and PyTorch; when to choose one.
  - Q2: Two use cases for Jupyter Notebooks in AI development.
  - Q3: How spaCy enhances NLP tasks vs. basic Python operations.
- **Comparative Analysis:** Scikit-learn vs. TensorFlow (target applications, ease of use, community support).

*Refer to `AI Tools and Applications.pdf` for detailed answers.*

### Part 2: Practical Implementation (50%)
- **Task 1: Classical ML with Scikit-learn (Iris Dataset)**
  - Preprocess data (handle missing values, encode labels).
  - Train a decision tree classifier for iris species prediction.
  - Evaluate with accuracy, precision, and recall.
  - Deliverable: `Task 1/Iris_Classification.py` or `Task 1/Iris_Classification.ipynb`.

- **Task 2: Deep Learning with TensorFlow/PyTorch (MNIST Dataset)**
  - Build a CNN model for handwritten digit classification (>95% test accuracy).
  - Visualize predictions on 5 sample images.
  - Deliverable: `Task 2/MNIST_CNN_Assignment.py` or `Task 2/MNIST_CNN_Assignment.ipynb`.

- **Task 3: NLP with spaCy (Amazon Product Reviews)**
  - Perform Named Entity Recognition (NER) for product names/brands.
  - Analyze sentiment (positive/negative) using rule-based approach.
  - Deliverable: `Task 3/npl_spacify_assignment.py` or `Task 3/npl_spacify_assignment.ipynb`.

### Part 3: Ethics & Optimization (10%)
- **Ethical Considerations:** Identify biases in MNIST or Amazon Reviews models; mitigation using TensorFlow Fairness Indicators or spaCy's rule-based systems.
- **Troubleshooting Challenge:** Debug and fix provided TensorFlow script (e.g., dimension mismatches, incorrect loss functions).

*Refer to `AI Tools and Applications.pdf` for ethical reflection and `Task 3/Ethical_Considerations.md` for details.*

### Bonus Task (Extra 10%)
- Deploy MNIST classifier using Streamlit or Flask.
- Deliverable: Web interface (e.g., `mnist_streamlit_app.py`), screenshot, and live demo link.

## Tools and Technologies
- **Frameworks:** TensorFlow, PyTorch, Scikit-learn, spaCy.
- **Platforms:** Google Colab (free GPU), Jupyter Notebook.
- **Datasets:** Iris (Kaggle/TensorFlow Datasets), MNIST (TensorFlow Datasets), Amazon Reviews (Kaggle).
- **Other:** Python 3.x, Pandas, NumPy, Matplotlib, Seaborn.

## Deliverables

### Code Implementation
- **Task 1 - Classical ML:** 
  - `Task 1/Iris_Classification.py` - Complete Scikit-learn implementation
  - `Task 1/Iris_Classification.ipynb` - Interactive Jupyter notebook version
  - `Task 1/Iris.csv` - Dataset file
- **Task 2 - Deep Learning:** 
  - `Task 2/MNIST_CNN_Assignment.py` - TensorFlow/Keras CNN implementation
  - `Task 2/MNIST_CNN_Assignment.ipynb` - Interactive Jupyter notebook version
  - `mnist_cnn_model.h5` - Trained model weights
- **Task 3 - NLP:** 
  - `Task 3/npl_spacify_assignment.py` - spaCy NER and sentiment analysis
  - `Task 3/npl_spacify_assignment.ipynb` - Interactive Jupyter notebook version
  - `Task 3/Ethical_Considerations.md` - Detailed ethical analysis
- **Bonus Task:** 
  - `mnist_streamlit_app.py` - Web deployment with drawing interface

### Documentation
- **`AI Tools and Applications.pdf`** - Comprehensive report containing:
  - Answers to theoretical questions (TensorFlow vs PyTorch, Jupyter use cases, spaCy advantages)
  - Screenshots of model outputs (accuracy graphs, confusion matrices, NER results)
  - Ethical reflection and bias mitigation strategies
  - Comparative analysis (Scikit-learn vs TensorFlow)
- **`README.md`** - This comprehensive project documentation
- **`TODO.md`** - Development progress tracking

### Additional Files
- **`AI Tools Assignment.docx`** - Original assignment guidelines
- **`~$ Tools Assignment.docx`** - Temporary Word document file

## Grading Rubric
- **Theoretical Accuracy:** 30%
- **Code Functionality & Quality:** 40%
- **Ethical Analysis:** 15%
- **Creativity & Presentation:** 15%

## Installation and Setup

### Prerequisites
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Git** for cloning the repository
- **Jupyter Notebook** or **JupyterLab** for interactive development
- **Virtual Environment** (recommended for dependency isolation)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd AI-Tools-Assignment
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv ai_tools_env

# Activate virtual environment
# On Windows:
ai_tools_env\Scripts\activate
# On macOS/Linux:
source ai_tools_env/bin/activate
```

### Step 3: Install Required Packages
```bash
# Core ML/DL libraries
pip install tensorflow==2.13.0
pip install torch torchvision torchaudio
pip install scikit-learn==1.3.0
pip install spacy==3.6.1

# Data manipulation and visualization
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Web deployment
pip install streamlit==1.25.0
pip install streamlit-drawable-canvas==0.9.3

# Additional utilities
pip install pillow==10.0.0
pip install jupyter==1.0.0
pip install notebook==6.5.4
```

### Step 4: Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Verify Installation
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
python -c "import spacy; print('spaCy version:', spacy.__version__)"
```

### Alternative: Using requirements.txt
```bash
# Create requirements.txt with all dependencies
pip install -r requirements.txt
```

## Usage Instructions

### Task 1: Iris Classification with Scikit-learn

**Objective:** Classify iris flower species using classical machine learning techniques.

**Dataset:** Iris dataset (150 samples, 4 features, 3 classes)

**Implementation Details:**
- Data preprocessing and exploration
- Decision Tree classifier training
- Model evaluation with accuracy, precision, and recall
- Visualization of results and confusion matrix

**How to Run:**
```bash
# Navigate to Task 1 directory
cd "Task 1"

# Run Python script
python Iris_Classification.py

# Or use Jupyter Notebook for interactive exploration
jupyter notebook Iris_Classification.ipynb
```

**Expected Output:**
- Dataset exploration and statistical summary
- Training and testing accuracy metrics
- Classification report with precision/recall for each species
- Confusion matrix visualization
- Feature importance analysis

### Task 2: MNIST CNN with TensorFlow/Keras

**Objective:** Build a Convolutional Neural Network to classify handwritten digits (0-9).

**Dataset:** MNIST dataset (70,000 grayscale images, 28x28 pixels)

**Model Architecture:**
- Convolutional layers with ReLU activation
- Max pooling for dimensionality reduction
- Dense layers for classification
- Dropout for regularization
- Target: >95% test accuracy

**How to Run:**
```bash
# Navigate to Task 2 directory
cd "Task 2"

# Run Python script (will train model and save weights)
python MNIST_CNN_Assignment.py

# Or use Jupyter Notebook for step-by-step exploration
jupyter notebook MNIST_CNN_Assignment.ipynb
```

**Expected Output:**
- Model architecture summary
- Training progress with loss/accuracy curves
- Test accuracy and evaluation metrics
- Predictions on sample images with visualizations
- Model weights saved as `mnist_cnn_model.h5`

### Task 3: NLP with spaCy

**Objective:** Perform Named Entity Recognition and sentiment analysis on Amazon product reviews.

**Features Implemented:**
- Named Entity Recognition for product names and brands
- Sentiment analysis using rule-based approach
- Text preprocessing and analysis
- Entity visualization and statistics

**How to Run:**
```bash
# Navigate to Task 3 directory
cd "Task 3"

# Run Python script
python npl_spacify_assignment.py

# Or use Jupyter Notebook for detailed analysis
jupyter notebook npl_spacify_assignment.ipynb
```

**Expected Output:**
- NER results showing extracted entities (PERSON, ORG, PRODUCT, etc.)
- Sentiment analysis results (positive/negative/neutral)
- Entity frequency analysis
- Sample text processing examples
- Performance metrics and evaluation

### Bonus Task: MNIST Streamlit Web App

**Objective:** Deploy the trained MNIST model as an interactive web application.

**Features:**
- Upload image files for digit prediction
- Interactive drawing canvas for digit input
- Real-time prediction with confidence scores
- User-friendly web interface

**How to Run:**
```bash
# From project root directory
streamlit run mnist_streamlit_app.py
```

**Access:** Open browser to `http://localhost:8501`

**Usage:**
1. **Upload Image:** Click "Choose an image" and select a PNG/JPG file
2. **Draw Digit:** Use the canvas to draw a digit (0-9)
3. **Get Prediction:** Click "Predict" to see the model's prediction
4. **Clear Canvas:** Use "Clear Canvas" to start over

**Requirements:**
- Trained model file (`mnist_cnn_model.h5`) must be in the root directory
- Images are automatically preprocessed (grayscale, 28x28 resize, normalization)

## Results and Performance Metrics

### Task 1: Iris Classification Results
- **Model:** Decision Tree Classifier
- **Training Accuracy:** ~100% (perfect fit on training data)
- **Test Accuracy:** ~95-98% (excellent generalization)
- **Precision:** High precision across all three species (Setosa, Versicolor, Virginica)
- **Recall:** Excellent recall rates for all classes
- **Key Insights:** 
  - Petal length and width are most important features
  - Clear separation between species in feature space
  - Minimal misclassification errors

### Task 2: MNIST CNN Results
- **Model:** Convolutional Neural Network
- **Architecture:** Conv2D → MaxPooling → Conv2D → MaxPooling → Dense → Dropout → Dense
- **Test Accuracy:** >95% (meets assignment requirement)
- **Training Time:** ~5-10 minutes on CPU, ~1-2 minutes on GPU
- **Model Size:** ~1.2MB (mnist_cnn_model.h5)
- **Key Insights:**
  - CNN architecture effectively captures spatial patterns
  - Dropout prevents overfitting
  - Model generalizes well to unseen handwritten digits

### Task 3: NLP Analysis Results
- **Named Entity Recognition:**
  - Successfully identifies PERSON, ORG, PRODUCT entities
  - High precision on brand names and product categories
  - Effective handling of various text formats
- **Sentiment Analysis:**
  - Rule-based approach provides consistent results
  - Good performance on clear positive/negative reviews
  - Challenges with neutral or mixed sentiment texts
- **Key Insights:**
  - spaCy's pre-trained models work well for general NER
  - Rule-based sentiment analysis suitable for structured reviews
  - Entity frequency analysis reveals popular products/brands

### Bonus Task: Streamlit App Performance
- **Deployment:** Successfully deployed as web application
- **User Interface:** Intuitive drawing canvas and file upload
- **Prediction Speed:** Real-time predictions (<1 second)
- **Accuracy:** Maintains model performance in web environment
- **User Experience:** Interactive and user-friendly interface

### Overall Project Success Metrics
- ✅ **All Tasks Completed:** 3/3 core tasks + 1 bonus task
- ✅ **Code Quality:** Well-documented, modular, and reusable
- ✅ **Performance Targets:** Met accuracy requirements for all models
- ✅ **Documentation:** Comprehensive README and code comments
- ✅ **Deployment:** Working web application demonstration

## Why This Matters
- **Real-World Impact:** Powers healthcare, finance, etc.
- **Skill Validation:** Proficiency in TensorFlow, PyTorch, Scikit-learn sought by employers.

## Ethical Considerations and Bias Mitigation

### MNIST Model Biases and Mitigation

**Identified Biases:**
- **Dataset Representation Bias:** MNIST may over-represent certain demographics (e.g., US postal workers) and under-represent others, leading to poorer performance on digits written by people from different cultural backgrounds or with disabilities
- **Model Fairness Bias:** CNN model might perform differently across subgroups, struggling with noisy or off-center inputs that might correlate with certain user groups
- **Evaluation Bias:** High overall accuracy might mask disparities in performance across specific digit classes or writing styles

**Mitigation Strategies:**
- **TensorFlow Fairness Indicators:** Compute metrics like demographic parity, equal opportunity, and disparate impact across subgroups
- **Bias Detection:** Use visualization tools to identify performance disparities and biased predictions
- **Model Adjustments:** Implement adversarial debiasing or reweighting samples to balance representation
- **Diverse Testing:** Test model on diverse datasets representing different writing styles and demographics

### Amazon Reviews Model Biases and Mitigation

**Identified Biases:**
- **Sentiment Analysis Bias:** Rule-based approaches (VADER/TextBlob) may not handle sarcasm, context, or cultural nuances well, leading to misclassification of positive words in negative contexts
- **NER Bias:** spaCy's en_core_web_sm model may not recognize product-specific entities well or have biases towards certain brands (over-recognition of major tech companies vs. lesser-known brands)
- **Data Bias:** Reviews might not represent all demographics equally, leading to biased sentiment towards products popular in certain groups

**Mitigation Strategies:**
- **Custom Rules for NER:** Define patterns to recognize underrepresented products or brands, reducing bias in entity extraction
- **Rule-Based Sentiment Adjustments:** Combine rule-based matching with ML to flag and correct biased sentiment predictions, including rules for sarcasm detection
- **Bias Audits:** Regularly audit the model on diverse datasets and adjust rules to ensure equitable performance across groups
- **Context-Aware Analysis:** Implement context-sensitive sentiment analysis to better handle nuanced language

### General Ethical Principles Applied

1. **Transparency:** All models and methodologies are clearly documented
2. **Fairness:** Regular bias testing and mitigation strategies implemented
3. **Accountability:** Clear responsibility for model decisions and outcomes
4. **Privacy:** Respect for user data and appropriate data handling practices
5. **Continuous Monitoring:** Ongoing evaluation of model performance and bias

### Responsible AI Development

This project demonstrates commitment to responsible AI development by:
- Identifying potential biases in each model
- Implementing appropriate mitigation strategies
- Documenting ethical considerations thoroughly
- Providing transparency in model decision-making processes
- Ensuring models are tested on diverse datasets

For detailed ethical analysis, refer to `Task 3/Ethical_Considerations.md`.

## Troubleshooting and FAQ

### Common Installation Issues

**Q: TensorFlow installation fails on Windows**
```bash
# Solution: Install Microsoft Visual C++ Redistributable
# Then try:
pip install tensorflow-cpu==2.13.0
# Or for GPU support:
pip install tensorflow-gpu==2.13.0
```

**Q: spaCy model download fails**
```bash
# Solution: Download manually
python -m spacy download en_core_web_sm --user
# Or try:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl
```

**Q: Streamlit canvas not working**
```bash
# Solution: Install specific version
pip install streamlit-drawable-canvas==0.9.3
# Clear browser cache and restart Streamlit
```

### Runtime Issues

**Q: MNIST model file not found**
- Ensure `mnist_cnn_model.h5` is in the root directory
- Run Task 2 first to generate the model file
- Check file permissions and path

**Q: Low accuracy on MNIST predictions**
- Verify image preprocessing (grayscale, 28x28, normalization)
- Check if uploaded image is clear and centered
- Try drawing digits more clearly on canvas

**Q: NER not detecting entities**
- Verify spaCy model is properly installed
- Check text preprocessing and encoding
- Try different text samples

### Performance Optimization

**Q: Training is too slow**
```bash
# Enable GPU acceleration (if available)
pip install tensorflow-gpu
# Or use Google Colab with free GPU
```

**Q: Memory issues with large datasets**
- Reduce batch size in model training
- Use data generators for large datasets
- Clear variables with `del` and `gc.collect()`

### File Path Issues

**Q: FileNotFoundError for datasets**
- Ensure Iris.csv is in `Task 1/` directory
- Check file paths use forward slashes `/` or raw strings `r"path"`
- Verify working directory is correct

**Q: Import errors**
```bash
# Solution: Check Python path and virtual environment
which python
pip list | grep tensorflow
# Reinstall if necessary:
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

### Streamlit App Issues

**Q: App won't start**
```bash
# Check if port 8501 is available
streamlit run mnist_streamlit_app.py --server.port 8502
# Or kill existing processes:
taskkill /f /im streamlit.exe  # Windows
pkill -f streamlit            # Linux/Mac
```

**Q: Canvas not responding**
- Refresh browser page
- Try different browser (Chrome recommended)
- Check browser console for JavaScript errors

### Debugging Tips

1. **Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check model summary:**
```python
model.summary()
model.get_config()
```

3. **Verify data shapes:**
```python
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

4. **Test individual components:**
```python
# Test data loading
df = pd.read_csv("Task 1/Iris.csv")
print(df.head())

# Test model loading
model = tf.keras.models.load_model('mnist_cnn_model.h5')
print("Model loaded successfully")
```

### Getting Help

- **Official Documentation:**
  - [TensorFlow](https://www.tensorflow.org/)
  - [PyTorch](https://pytorch.org/)
  - [spaCy](https://spacy.io/)
  - [Streamlit](https://docs.streamlit.io/)

- **Community Support:**
  - Post on LMS Community with #AIToolsAssignment
  - Stack Overflow for specific technical issues
  - GitHub Issues for project-specific problems

- **Pro Tips:**
  - Test code incrementally for small wins
  - Use print statements for debugging
  - Save model checkpoints during training
  - Document any custom modifications

## Project Structure

```
AI-Tools-Assignment/
├── README.md                           # This comprehensive documentation
├── AI Tools Assignment.docx            # Original assignment guidelines
├── AI Tools and Applications.pdf       # Detailed report with answers and screenshots
├── mnist_cnn_model.h5                  # Trained MNIST model weights
├── mnist_streamlit_app.py              # Web deployment application
├── TODO.md                             # Development progress tracking
├── Task 1/                             # Classical ML with Scikit-learn
│   ├── Iris_Classification.py         # Python implementation
│   ├── Iris_Classification.ipynb       # Jupyter notebook
│   └── Iris.csv                        # Dataset
├── Task 2/                             # Deep Learning with TensorFlow
│   ├── MNIST_CNN_Assignment.py         # Python implementation
│   └── MNIST_CNN_Assignment.ipynb     # Jupyter notebook
└── Task 3/                             # NLP with spaCy
    ├── npl_spacify_assignment.py       # Python implementation
    ├── npl_spacify_assignment.ipynb    # Jupyter notebook
    └── Ethical_Considerations.md       # Detailed ethical analysis
```

## Key Features Demonstrated

### Technical Skills
- **Machine Learning:** Scikit-learn for classical ML tasks
- **Deep Learning:** TensorFlow/Keras for CNN implementation
- **Natural Language Processing:** spaCy for NER and sentiment analysis
- **Web Development:** Streamlit for model deployment
- **Data Visualization:** Matplotlib and Seaborn for insights
- **Model Evaluation:** Comprehensive metrics and analysis

### Best Practices
- **Code Documentation:** Well-commented and modular code
- **Version Control:** Git repository with clear commit history
- **Testing:** Incremental testing and validation
- **Deployment:** Production-ready web application
- **Ethics:** Bias identification and mitigation strategies

## Future Enhancements

- **Model Improvements:** Hyperparameter tuning and advanced architectures
- **Data Augmentation:** Enhanced training data for better generalization
- **Real-time Processing:** Live data streaming and processing
- **Mobile Deployment:** Mobile app versions of the models
- **Advanced NLP:** Transformer models for improved text analysis
- **Monitoring:** Model performance tracking and alerting systems

## Group Members
- **Peter Mwaura** - Project Lead and Implementation
- [Add other group members as applicable]

## License
This project is for educational purposes as part of the AI Tools and Applications assignment. Refer to assignment guidelines for usage and distribution policies.

---

**🎯 Assignment Completion Status:**
- ✅ Task 1: Iris Classification (Scikit-learn)
- ✅ Task 2: MNIST CNN (TensorFlow/Keras) 
- ✅ Task 3: NLP Analysis (spaCy)
- ✅ Bonus: Streamlit Web App Deployment
- ✅ Ethical Considerations Documentation
- ✅ Comprehensive README Documentation

**📊 Final Grade Components:**
- Theoretical Understanding: 40%
- Practical Implementation: 50% 
- Ethics & Optimization: 10%
- Bonus Task: +10%

*This README serves as comprehensive documentation for the AI Tools and Applications assignment, demonstrating mastery of modern AI frameworks and responsible AI development practices.*
