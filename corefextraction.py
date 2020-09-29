# -*- coding: utf-8 -*-
import re
import torch
import spacy
import neuralcoref
from allennlp.predictors.predictor import Predictor
from nltk.tokenize import sent_tokenize, word_tokenize


class Tripple:
	def __init__(self, arg0,verb,arg1):
		self.subject = arg0.lower()
		self.verb = verb.lower()
		self.object = arg1.lower()
	def __str__(self):
		return f'{self.subject.capitalize()} {self.verb} {self.object}. '

	def __repr__(self):
		return f'Tripple({self.subject!r},{self.verb!r},{self.object!r})'
	def __eq__(self, other):
		return ((self.verb in other.verb) and (other.object in self.object)) or ((other.verb in self.verb) and (self.object in other.object))
	def __len__(self):
		return len(str(self))


class InformationExtractor():
	def __init__(self, coreference = False):

		self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
		if torch.cuda.is_available():
			self.predictor._model = self.predictor._model.cuda(0)

		self.spacy_pipeline = spacy.load('en')
		self.coreference = coreference
		if self.coreference:
			coref = neuralcoref.NeuralCoref(self.spacy_pipeline.vocab)
			self.spacy_pipeline.add_pipe(coref, name='neuralcoref')

	def Arguments(self):
		_dict = dict({})
		def dict_instance(string):
			values = string.split(': ')
			if len(values) > 1:
				_dict[values[0]] = values[1]
			return _dict
		return dict_instance

	def find_tripples(self,string):
		tripples = []
		extraction = self.predictor.predict(
	 	 sentence=string
		)
		#print(extraction)
		for phrase in extraction['verbs']:
			args = dict({})
			subject = None
			action = None
			object1 = None
			object2 = None
			matches=re.findall(r'\[(.+?)\]',phrase['description'])
			for x in matches:
				keyValues = x.split(': ')
				if len(keyValues) > 1:
					args[keyValues[0]] = keyValues[1]
			if 'ARG0' in args:
				subject = args['ARG0']
			if 'ARG1' in args:
				object1 = args['ARG1']
			if 'ARG2' in args:
				if object1 is not None:
					object1 = object1 + ' ' + args['ARG2']
				else:
					object1 = args['ARG2']
			if 'V' in args:
				action = args['V']
			if 'BV' in args:
				action = args['BV'] +' '+action
			if 'AV' in args:
				action = action + ' ' + args['AV']

			if subject and action and object1:
				new_tripple = Tripple(subject,action,object1)
				#tripples.append(new_tripple)
				#print(new_tripple)
				if len(tripples):
					old_tripple = tripples[-1]
					if old_tripple == new_tripple:
						if len(new_tripple.verb) > len(old_tripple.verb):
							tripples[-1] = new_tripple
					else:
						tripples.append(new_tripple)
				else:
					tripples.append(new_tripple)

		#if tripples:
		#	return max(tripples, key=len)
		#else:
		#	return None
		return tripples


	def process(self,text):
		sentnces = self.sent_tokenize(text)
		tripples = [self.find_tripples(sent) for sent in sentnces]
		tripples =[sent for sent in tripples if sent is not None]
		output = []
		for tripple in tripples:
			output += tripple

		return output


	def sent_tokenize(self,input_):
		if not self.coreference:
			if isinstance(input_,list):
				sentences = input_
			else:
				document = self.spacy_pipeline(input_)
				sentences = [str(sent) for sent in document.sents]
		else:
			if isinstance(input_,list):
				document = self.spacy_pipeline(" ".join(input_))
				sentences = input_
			else:
				document = self.spacy_pipeline(input_)
				sentences = [str(sent) for sent in document.sents]

			if document._.has_coref:
				sentences = self.get_resolved(document, sentences)

		output = sentences
		return output

	def get_resolved(self, doc, sentences):
		def get_2d_element(arrays, index):
			j = index
			lens = [len(sent) for sent in arrays]
			for i,length in enumerate(lens):
				j = j - length
				if j < 0:
					return i, length + j
		resolved_list = []
		tokenizer = spacy.load('en')
		for sent in sentences:
			resolved_list.append(list(tok.text_with_ws for tok in tokenizer(sent)))

		for cluster in doc._.coref_clusters:
			for coref in cluster:
				if coref != cluster.main:
					ind1, ind2 = get_2d_element(resolved_list,coref.start)
					resolved_list[ind1][ind2] = cluster.main.text + doc[coref.end-1].whitespace_
					for i in range(coref.start+1, coref.end):
						ind3, ind4 = get_2d_element(resolved_list,i)
						resolved_list[ind3][ind4] = ""
		output = [''.join(sublist) for sublist in resolved_list]
		return output