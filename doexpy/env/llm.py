from doexpy.env.discrete_env import DiscreteEnv
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from typing import List
import torch


class LLMGrid(DiscreteEnv):
	def __init__(
		self,
		list_of_text_tokens: List[str] = None,
		device: str = 'cpu',
		MODELS_CACHE_DIR: str = '/tmp/models_cache_dir/',
		verbose: bool = False,
		normalize: bool = True
	):
		self.verbose = verbose
		self.device = device
		self.constrained = False
		super().__init__(
			init_state=0,
		)
		self.normalize = normalize
		self.list_of_text_tokens = list_of_text_tokens
		self.max_episode_length = len(list_of_text_tokens)

		# collapse the text tokens into a single list
		self.tokens = {}

		index = 1
		self.tokens[' '] = [i for i in range(self.max_episode_length)]
		self.unique_elements = [' ']
		for order, list in enumerate(list_of_text_tokens):
			for token in list:
				if token not in self.tokens:
					self.unique_elements.append(token)
					self.tokens[token] = [order]
					index += 1
				else:
					self.tokens[token] += [order]

		# get the total number of states
		total_tokens = len(self.unique_elements)

		# create a LLM embedding model CLIP
		clip_model_id = "openai/clip-vit-large-patch14"  # All stable diffusion models rely on VIT-14
		self._tokenizer = CLIPTokenizer.from_pretrained(clip_model_id, cache_dir=str(MODELS_CACHE_DIR))
		self._processor = CLIPProcessor.from_pretrained(clip_model_id, cache_dir=str(MODELS_CACHE_DIR))
		self._model = CLIPModel.from_pretrained(clip_model_id, cache_dir=str(MODELS_CACHE_DIR)).to(self.device)

		self.states_num = self.max_episode_length
		self.actions_num = total_tokens
		self.h = 0
		self.action_space_pre_embedding = torch.arange(self.actions_num, dtype=torch.float64).reshape(-1, 1)
		self.emiss_num = self.actions_num
		self.transition_matrix = None
		self._generate_emissions()

		self.action_space = self.emissions
		self.visitations = torch.zeros(self.states_num, self.actions_num, dtype=torch.float64)


	def _em(self, text: str):
		text_input = self._tokenizer(
						text,
						padding="max_length",
						max_length=self._tokenizer.model_max_length,
						truncation=True,
						return_tensors="pt",
					)
		feat = self._model.get_text_features(**text_input)  # projected CLIP embeddings
		feat = feat.detach().double()
		if self.normalize:
			feat = feat / torch.norm(feat, p=2)
		return feat
	
	def embed_text(
		self,
		list_of_texts: str
	) -> torch.Tensor:
		
		emissions = []
		for text in list_of_texts:
			feat = self._em(text)
			emissions.append(feat)
		emissions = torch.vstack(emissions)
		return emissions

	def embed_action_clip(
		self,
		actions: List
	) -> torch.Tensor:
		"""
		Embeds a list of actions using the CLIP model.
		"""
		emissions = []
		for i in actions:
			text = self.unique_elements[i]
			feat = self._em(text)
			emissions.append(feat)
		emissions = torch.vstack(emissions)
		return emissions

	def _generate_emissions(self):
		"""
		Generates the emissions for the environment.
		"""
		if self.verbose:
			print("PREPROCESS: Generating emissions")
		self.emissions = []
		actions = list(range(self.actions_num)) 
		self.emissions = self.embed_action_clip(actions)
		
		# for i in range(self.actions_num):

		#     text = self.unique_elements[i]
		#     text_input = self._tokenizer(
		#         text,
		#         padding="max_length",
		#         max_length=self._tokenizer.model_max_length,
		#         truncation=True,
		#         return_tensors="pt",
		#     )
		#     if self.verbose:
		#         print(f"Generating emission for action {i}, text: {text}")
		#     feat = self._model.get_text_features(**text_input)  # projected CLIP embeddings
		#     feat = feat.detach().double()
		#     self.emissions.append(feat)
		# if self.verbose:
		#     print("Done generating.")
		# self.emissions = torch.vstack(self.emissions)

	def next(self, state, action):
		return state + 1

	def convert(self, state):
		pass

	def available_actions(self, state):
		actions = []
		actions.append(0)
		for i in range(1, self.actions_num):
			if self.is_valid_action(i, state):
				actions.append(i)
		return actions

	def step(self, action):
		self.visitations[self.state, action] += 1
		self.state = self.next(self.state, action)
		return self.state

	def is_valid_action(self, action, state) -> bool:
		if action == 0 and state > 0:
			return True
		else:
			if state in self.tokens[self.unique_elements[action]]:
				return True
			else:
				return False

	def p_next(self, state, action):
		probs = {min(state + 1, self.max_episode_length - 1): 1}
		return probs

	def get_transition_matrix(self) -> torch.Tensor:
		if self.transition_matrix is not None:
			return self.transition_matrix

		P = torch.zeros(size=(self.states_num, self.actions_num, self.states_num), dtype=torch.float64)
		for s in range(self.states_num):
			for a in range(self.actions_num):
				if self.is_valid_action(a, s):
					if self.verbose:
						print(f"Valid action {a} in state {s}")
					probs = self.p_next(s, a)
					for s_state in probs.keys():
						P[s, a, s_state] = probs[s_state]
		self.transition_matrix = P
		return P

	def reset(self) -> None:
		self.state = self.init_state
		self.h = 0


if __name__ == "__main__":
	file_path = '../../experiments/llm/mediums.txt'
	with open(file_path, 'r') as file:
		words_list = [line.strip() for line in file]
	env = LLMGrid(list_of_text_tokens=[words_list, words_list], verbose=True)
