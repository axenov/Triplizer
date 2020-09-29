# Triplizer

It is a  simple wrapper for the [AllenNLP](https://github.com/allenai/allennlp) Open Information Extrction model and the Coreference Resolution library [NeuralCoref](https://github.com/huggingface/neuralcoref). It extracts from a text a list of tripples in the form of set (Subject, Action, Object).

The example of usage:
```python
from corefextraction import InformationExtractor
extractor = InformationExtractor(coreference = True)
text = "Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye Allen. Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two years younger, with whom he shared an enthusiasm for computers."
triples = extractor.process(text)
for triple in triples:
	print(triple)
'''
Output:
Paul allen was born on january 21 , 1953. 
Paul allen attended lakeside school. 
Paul allen befriended bill gates. 
Paul allen shared an enthusiasm for computers.
'''
```
