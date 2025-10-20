# Amazon Product Reviews NLP Analysis with spaCy
# NER and Sentiment Analysis

import spacy
import pandas as pd
import nltk
from spacy import displacy
from spacy.matcher import Matcher
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data (run once)
try:
    nltk.download('vader_lexicon')
except:
    print("NLTK data already downloaded or download failed")

class AmazonReviewAnalyzer:
    def __init__(self):
        """Initialize the analyzer with spaCy model and sentiment analyzer"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully")
        except OSError:
            print("Please download spaCy model first: python -m spacy download en_core_web_sm")
            return
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self._setup_matcher()
    
    def _setup_matcher(self):
        """Setup pattern matcher for enhanced product recognition"""
        self.matcher = Matcher(self.nlp.vocab)
        
        # Define patterns for common product types in Amazon reviews
        product_patterns = [
            [{"LOWER": "iphone"}], [{"LOWER": "macbook"}], 
            [{"LOWER": "ipad"}], [{"LOWER": "airpods"}],
            [{"LOWER": "kindle"}], [{"LOWER": "echo"}],
            [{"LOWER": "fire"}, {"LOWER": "tv"}],
            [{"LOWER": "galaxy"}, {"LOWER": "phone"}],
            [{"LOWER": "galaxy"}, {"LOWER": "s"}],
            [{"LOWER": "pixel"}, {"LOWER": "phone"}],
            [{"LOWER": "playstation"}], [{"LOWER": "xbox"}],
            [{"LOWER": "nintendo"}, {"LOWER": "switch"}]
        ]
        
        self.matcher.add("PRODUCT_TERMS", product_patterns)
    
    def extract_entities(self, review_text):
        """
        Extract product names and brands from review text using spaCy NER
        """
        doc = self.nlp(review_text)
        entities = []
        
        # Extract entities using spaCy's built-in NER
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        return entities
    
    def extract_entities_enhanced(self, review_text):
        """
        Enhanced entity extraction combining spaCy NER and pattern matching
        """
        doc = self.nlp(review_text)
        entities = []
        
        # Use pattern matcher for product recognition
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            entities.append({
                'text': span.text,
                'label': 'PRODUCT',
                'start': start,
                'end': end
            })
        
        # Add spaCy's built-in entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
                # Avoid duplicates
                if not any(e['text'] == ent.text and e['label'] == ent.label_ for e in entities):
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
        
        return entities
    
    def analyze_sentiment_textblob(self, review_text):
        """
        Analyze sentiment using TextBlob (rule-based)
        """
        blob = TextBlob(review_text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "POSITIVE"
        elif polarity < -0.1:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_sentiment_vader(self, review_text):
        """
        Analyze sentiment using VADER (specifically designed for reviews)
        """
        scores = self.sentiment_analyzer.polarity_scores(review_text)
        compound_score = scores['compound']
        
        if compound_score >= 0.05:
            sentiment = "POSITIVE"
        elif compound_score <= -0.05:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            'sentiment': sentiment,
            'compound_score': compound_score,
            'scores': scores
        }
    
    def analyze_review_complete(self, review_text, use_enhanced_ner=True):
        """
        Complete analysis: NER + Sentiment Analysis
        """
        # Extract entities
        if use_enhanced_ner:
            entities = self.extract_entities_enhanced(review_text)
        else:
            entities = self.extract_entities(review_text)
        
        # Analyze sentiment (using VADER for better performance on reviews)
        sentiment_result = self.analyze_sentiment_vader(review_text)
        
        return {
            'review_text': review_text,
            'entities': entities,
            'sentiment': sentiment_result
        }
    
    def visualize_entities(self, review_text):
        """
        Visualize named entities in the review text
        """
        doc = self.nlp(review_text)
        displacy.render(doc, style="ent", jupyter=False)
    
    def analyze_multiple_reviews(self, reviews):
        """
        Analyze multiple reviews and return results as DataFrame
        """
        results = []
        for review in reviews:
            analysis = self.analyze_review_complete(review)
            results.append({
                'review': review,
                'sentiment': analysis['sentiment']['sentiment'],
                'sentiment_score': analysis['sentiment']['compound_score'],
                'entities': [f"{ent['text']} ({ent['label']})" for ent in analysis['entities']],
                'entity_count': len(analysis['entities'])
            })
        
        return pd.DataFrame(results)

def main():
    """
    Main function to demonstrate the analysis
    """
    print("Amazon Product Reviews NLP Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AmazonReviewAnalyzer()
    
    # Sample Amazon reviews for demonstration
    sample_reviews = [
        "The Apple iPhone 15 has an amazing camera but the battery life is terrible. I expected better from Apple.",
        "Samsung Galaxy S23 is fantastic! The display is brilliant and the performance is smooth. Much better than Google Pixel.",
        "I love my new Kindle Paperwhite for reading books at night! The backlight is perfect.",
        "The Sony headphones broke after just two weeks. Very disappointed with the quality.",
        "Microsoft Surface Pro is a great device for work, but the price is too high compared to Apple iPad.",
        "Bought this Amazon Echo Dot and it's been working perfectly. Alexa understands all my commands.",
        "The battery on this Nintendo Switch doesn't last long. Otherwise, it's a good gaming console.",
        "Google Pixel camera is outstanding but the software has too many bugs.",
        "MacBook Air is lightweight and fast, perfect for students and professionals.",
        "The Dell laptop overheats constantly and the customer service was unhelpful."
    ]
    
    print(f"Analyzing {len(sample_reviews)} sample reviews...")
    print()
    
    # Analyze each review
    for i, review in enumerate(sample_reviews, 1):
        print(f"REVIEW {i}:")
        print(f"Text: {review}")
        
        # Perform complete analysis
        result = analyzer.analyze_review_complete(review)
        
        # Print sentiment results
        sentiment = result['sentiment']
        print(f"Sentiment: {sentiment['sentiment']} (Score: {sentiment['compound_score']:.3f})")
        
        # Print entities
        entities = result['entities']
        if entities:
            print("Entities Found:")
            for entity in entities:
                print(f"  - {entity['text']} ({entity['label']})")
        else:
            print("No relevant entities found.")
        
        print("-" * 80)
        print()
    
    # Create summary analysis
    print("SUMMARY ANALYSIS")
    print("=" * 50)
    
    df_results = analyzer.analyze_multiple_reviews(sample_reviews)
    
    # Print summary statistics
    print("\nSentiment Distribution:")
    sentiment_counts = df_results['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} reviews")
    
    print(f"\nTotal Entities Found: {df_results['entity_count'].sum()}")
    
    # Show all results in a table
    print("\nDetailed Results Table:")
    print(df_results[['review', 'sentiment', 'sentiment_score', 'entities']].to_string(index=False))
    
    # Example of entity visualization for one review
    print("\n" + "="*50)
    print("ENTITY VISUALIZATION EXAMPLE")
    print("="*50)
    example_review = "Apple iPhone and Samsung Galaxy are both great phones, but I prefer Google Pixel for its camera."
    print(f"Review: {example_review}")
    print("\nEntity visualization (run in Jupyter for better display):")
    
    # For non-Jupyter environments, we'll print the entities manually
    doc = analyzer.nlp(example_review)
    print("Entities found:")
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
            print(f"  {ent.text} - {ent.label_}")

if __name__ == "__main__":
    main()