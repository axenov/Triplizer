# Triplizer

```python
from corefextraction import InformationExtractor
extractor = InformationExtractor(coreference = True)
text = "Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye Allen. 
	Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two years younger,
	with whom he shared an enthusiasm for computers."
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
