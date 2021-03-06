# Sentimientos
A free python sentiment analysis library for Spanish.

# Description
The model is based on a bidirectional recurrent neural network. It has been trained on a corpus of ~70k reviews collected from trustpilot using web scraping tecniques and ~150k neutral sentences collected from Wikipedia.

# Usage
Install sentimientos using pip:

```bash
pip install git+https://github.com/AlessandroGianfelici/sentimientos.git
```

Download the SpaCy model for Spanish:

```bash
python -m spacy download es_core_news_sm
```

Try some sample sentences:

```python
from sentimientos import calculate_polarity

calculate_polarity(["sample sentence 1", "sample sentence 2", "sample sentence 3"], verbose=True)
```

the result are the polarities (the first component is the positive polarity and the second component is the negative polarity) of the sentences.

# Credits
The model is heavily based on SentITA, an analogous library for the italian language:

https://github.com/NicGian/SentITA

Sentimientos uses the same network architecture, but a quite different preprocessing of the input texts.
