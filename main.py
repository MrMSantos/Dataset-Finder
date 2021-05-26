import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from nltk.tokenize import sent_tokenize

import torch
from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import DistilBertTokenizerFast, BertTokenizerFast
from transformers import DistilBertForQuestionAnswering, DistilBertForTokenClassification, BertForTokenClassification
from transformers import pipeline

def clean_text(txt):
	return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

def get_full_publication(path):

	complete_publication = ''
	with open(path, 'r') as f:
		json_decode = json.load(f)

		for data in json_decode:
			section_title = data.get('section_title')
			text = data.get('text')
	
			complete_publication += section_title + '. ' + text + ' '

	return complete_publication

def read_to_qa(train_csv):

	contexts = []
	questions = []
	answers = []
	question = 'What is the dataset used in this publication?'
	CHAR_PADDING = 1024

	for index, row in train_csv.iterrows():
		combined = []
		answer = train_csv.iloc[index]['dataset_label']
		with open('./train/' + train_csv.iloc[index]['Id'] + '.json', 'r') as f:
			full_pub = json.load(f)

			for section in full_pub:
				text = section.get('text')
				
				if answer in text:
					section_title = section.get('section_title')
					context = section_title + '. ' + text
					context = context.replace('\n', ' ')
					context = context.replace('\t', ' ')
					context_start = context.find(answer) - CHAR_PADDING
					context_end = context.find(answer) + CHAR_PADDING
					if context_start < 0:
						context = context[:context_end + abs(context_start)]
					else:
						context = context[context_start:context_end]

					contexts.append(context)
					answers.append({'text': answer, 'answer_start': context.find(answer), 'answer_end': context.find(answer) + len(answer) - 1}) #FIXED
					questions.append(question)

	return contexts, questions, answers

def read_to_ner(train_csv):

	pos_sentences = []
	neg_sentences = []
	pos_tags = []
	neg_tags = []
	for index, row in train_csv.iterrows():
		combined = []
		answer = train_csv.iloc[index]['dataset_label']
		path = './train/' + train_csv.iloc[index]['Id'] + '.json'
		pub_string = get_full_publication(path)

		answer_words = answer.split()
		
		sentences = sent_tokenize(pub_string)
		short_sentences = shorten_sentences(sentences)

		for sentence in short_sentences:
			sentence_words = sentence.split()

			key_index = find_sublist(sentence_words, answer_words)
			if key_index != None:
				pos_sentences.append(sentence_words)
				tags = ['O'] * len(sentence_words)
				tags[key_index] = 'B'
				for i in range(key_index + 1, key_index + len(answer_words)):
					tags[i] = 'I'
				pos_tags.append(tags)
			elif any(word in sentence.lower() for word in ['dataset']):#upper_sentence(sentence_words):
				neg_sentences.append(sentence_words)
				tags = ['O'] * len(sentence_words)
				neg_tags.append(tags)

	#print(len(pos_sentences))
	#print(len(neg_sentences))

	word_sentences = pos_sentences + neg_sentences
	word_tags = pos_tags + neg_tags

	return word_sentences, word_tags

def find_sublist(big_list, small_list):

	position = None
	for i in range(len(big_list) - len(small_list) + 1):
		if small_list == big_list[i:i + len(small_list)]:
			position = i
			return position

def upper_sentence(sentence):

	for i in range(len(sentence) - 2):
		if sentence[i][0].isupper() and sentence[i + 1][0].isupper() and sentence[i + 2][0].isupper():
			return True

def encode_tags(tags, encodings, label2id):

	labels = [[label2id[tag] for tag in doc] for doc in tags]
	encoded_labels = []

	for i, label in enumerate(labels):
		word_ids = encodings.word_ids(batch_index = i)
		previous_word_idx = None
		label_ids = []
		for word_idx in word_ids:
			# Special tokens have a word id that is None. We set the label to -100 so they are automatically
			# ignored in the loss function.
			if word_idx is None:
				label_ids.append(-100)
			# We set the label for the first token of each word.
			elif word_idx != previous_word_idx:
				label_ids.append(label[word_idx])
			# For the other tokens in a word, we set the label to either the current label or -100, depending on
			# the label_all_tokens flag.
			else:
				label_ids.append(-100)
			previous_word_idx = word_idx

		encoded_labels.append(label_ids)

	return encoded_labels

def shorten_sentences(sentences, max_length = 64, overlap = 20):
	
	short_sentences = []
	for sentence in sentences:
		words = sentence.split()
		if len(words) > max_length:
			for p in range(0, len(words), max_length - overlap):
				short_sentences.append(' '.join(words[p:p + max_length]))
		else:
			short_sentences.append(sentence)

	return short_sentences

def add_token_positions(encodings, answers):

	start_positions = []
	end_positions = []

	for i in range(len(answers)):
		start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
		end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

		#if start position is None, the answer passage has been truncated
		if start_positions[-1] is None:
			start_positions[-1] = tokenizer.model_max_length
		
		if end_positions[-1] is None:
			end_positions[-1] = tokenizer.model_max_length

	encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class QADataset(torch.utils.data.Dataset):

	def __init__(self, encodings):
		self.encodings = encodings

	def __getitem__(self, idx):
		return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

	def __len__(self):
		return len(self.encodings.input_ids)

class NERDataset(torch.utils.data.Dataset):

	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)

def qa_training(train_csv, model, tokenizer, epochs = 2, batch_size = 8):

	#get the lists of contents
	contexts, questions, answers = read_to_qa(train_csv)

	train_encodings = tokenizer(contexts, questions, truncation = True, padding = True)
	add_token_positions(train_encodings, answers)
	train_dataset = QADataset(train_encodings)
	
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)
	model.train()

	train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

	optim = AdamW(model.parameters(), lr = 5e-5)

	for epoch in range(epochs):
		with tqdm(train_loader, unit = 'batch') as tepoch:
			for batch in tepoch:
			#for batch in train_loader:
				optim.zero_grad()
				input_ids = batch['input_ids'].to(device)
				attention_mask = batch['attention_mask'].to(device)
				start_positions = batch['start_positions'].to(device)
				end_positions = batch['end_positions'].to(device)
				outputs = model(input_ids, attention_mask = attention_mask, start_positions = start_positions, end_positions = end_positions)
				loss = outputs[0]
				loss.backward()
				optim.step()

	model.eval()
	model.save_pretrained('./distilbert_finetuned_qa_2e_squad')

def ner_training(train_csv, model, tokenizer, label2id, epochs = 2, batch_size = 32):

	#get the lists of sentences and tags
	train_texts, train_tags = read_to_ner(train_csv)

	#get unique labels for the tags
	#unique_tags = set(tag for doc in train_tags for tag in doc)
	#tag2id = {tag: id for id, tag in enumerate(unique_tags)}
	#id2tag = {id: tag for tag, id in tag2id.items()}

	train_encodings = tokenizer(train_texts, is_split_into_words = True, return_offsets_mapping = True, padding = True, truncation = True, max_length = 128)
	train_labels = encode_tags(train_tags, train_encodings, label2id)

	train_encodings.pop('offset_mapping')
	train_dataset = NERDataset(train_encodings, train_labels)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)
	model.train()

	train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

	optim = AdamW(model.parameters(), lr = 5e-5)

	for epoch in range(epochs):
		with tqdm(train_loader, unit = 'batch') as tepoch:
			for batch in tepoch:
			#for batch in train_loader:
				optim.zero_grad()
				input_ids = batch['input_ids'].to(device)
				attention_mask = batch['attention_mask'].to(device)
				labels = batch['labels'].to(device)
				outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
				loss = outputs[0]
				loss.backward()
				optim.step()

	model.eval()
	model.save_pretrained('./bert_finetuned_ner_2e')

def qa_inference(model, tokenizer, char_padding = 2048, char_window = 256, threshold = 0.80):

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)
	model.eval()

	question = 'What is the dataset used in this publication?'
	pub_ids = []
	answers = []

	softmax = torch.nn.Softmax(dim = 1)

	for file in os.listdir('./test/'):
		path = './test/' + file
		text = get_full_publication(path)
		pub_answers = []

		for i in range(0, len(text), char_padding):
			if i == 0:
				input_dict = tokenizer(question, text[i:i + char_padding], return_tensors = 'pt', truncation = True, padding = True)
			else:
				input_dict = tokenizer(question, text[i - char_window:i + char_padding - char_window], return_tensors = 'pt', \
					truncation = True, padding = True)

			input_dict.to(device)
			outputs = model(**input_dict)
			start_logits = outputs.start_logits
			end_logits = outputs.end_logits

			start_logits = softmax(start_logits)
			end_logits = softmax(end_logits)

			prob_start = torch.max(start_logits)
			prob_end = torch.max(end_logits)

			all_tokens = tokenizer.convert_ids_to_tokens(input_dict['input_ids'][0])
			answer = tokenizer.convert_tokens_to_string(all_tokens[torch.argmax(start_logits, 1)[0] : torch.argmax(end_logits, 1)[0] + 1])

			if prob_start > threshold and prob_end > threshold:
				if answer not in pub_answers:
					answer = clean_text(answer)
					pub_answers.append(answer)
					print(prob_start)
					print(prob_end)
					print(answer)

		pub_ids.append(file[:-5])
		answers.append('|'.join(pub_answers))

	submission = pd.DataFrame()
	submission['Id'] = pub_ids
	submission['PredictionString'] = answers

	#print(submission.head())
	#submission.to_csv('submission.csv', index = False)

def ner_inference(model, tokenizer, threshold = 0.50):

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)
	model.eval()

	pub_ids = []
	answers = []

	softmax = torch.nn.Softmax(dim = 1)

	for file in os.listdir('./test/'):
		path = './test/' + file
		text = get_full_publication(path)
		sentences = sent_tokenize(text)
		short_sentences = shorten_sentences(sentences)
		pub_answers = []

		for sentence in short_sentences:
			input_dict = tokenizer(sentence, return_tensors = 'pt', truncation = True, padding = True, max_length = 128)
			input_dict.to(device)
			output = model(**input_dict)
			logits = output.logits
			logits = softmax(logits[0])
			ids_index = []
			for i, token_logits in enumerate(logits):
				if len(ids_index) == 0 and token_logits[model.config.label2id['B']] > threshold:
					ids_index.append(i)
				elif len(ids_index) > 0 and token_logits[model.config.label2id['I']] > threshold:
					ids_index.append(i)

			if len(ids_index) > 0:
				ids_list = input_dict['input_ids'][0][ids_index[0]:ids_index[-1] + 1]
				tokens = tokenizer.convert_ids_to_tokens(ids_list)
				answer = tokenizer.convert_tokens_to_string(tokens)
				answer = clean_text(answer)

				if answer not in pub_answers:
					pub_answers.append(answer)
					print(answer)

		pub_ids.append(file[:-5])
		answers.append('|'.join(pub_answers))

	submission = pd.DataFrame()
	submission['Id'] = pub_ids
	submission['PredictionString'] = answers

	print(answers)
	print(submission.head())
	#submission.to_csv('submission.csv', index = False)



if __name__ == '__main__':
	
	#load train target values
	train_csv = pd.read_csv('./train.csv')

	'''
	#QA TRAINING
	#load BERT tokenizer and BERT model for QA
	tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-distilled-squad')
	#model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

	#qa_training(train_csv, model, tokenizer, epochs = 2, batch_size = 8))

	#QA INFERENCE
	model = DistilBertForQuestionAnswering.from_pretrained('./distilbert_finetuned_qa_2e_squad')
	qa_inference(model, tokenizer, char_padding = 2048, char_window = 256, threshold = 0.80)

	'''
	#NER TRAINING
	#load BERT tokenizer and BERT model for NER
	label2id = {'O': 0, 'B': 1, 'I': 2}
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
	#model = BertForTokenClassification.from_pretrained('bert-base-cased', label2id = label2id, num_labels = len(label2id))

	#ner_training(train_csv, model, tokenizer, label2id, epochs = 2, batch_size = 32)

	#NER INFERENCE
	model = BertForTokenClassification.from_pretrained('./bert_finetuned_ner_2e', label2id = label2id)
	ner_inference(model, tokenizer, threshold = 0.70)
	