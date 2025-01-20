from doexpy.env.discrete_env import DiscreteEnv
from image_generator import StableDiffusionGenerator
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from typing import List, Tuple, Union
import torch
from torch import nn
import os
import hashlib
import pickle


class LLMGrid(DiscreteEnv):
    def __init__(
        self,
        list_of_text_tokens: List[str],
        model: CLIPModel,
        processor: CLIPProcessor, 
        tokenizer: CLIPTokenizer,
        cache_dir: str,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.constrained = False
        super().__init__(init_state=0)
        
        self.device = next(model.parameters()).device
        self.embedder = CLIPEmbedder(tokenizer, model)
        self._processor = processor
        self._tokenizer = tokenizer

        self.cache_dir = cache_dir

        self.list_of_text_tokens = list_of_text_tokens
        self.max_episode_length = len(list_of_text_tokens)

        # Setup tokens dictionary
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

        total_tokens = len(self.unique_elements)
        
        self.states_num = self.max_episode_length
        self.actions_num = total_tokens
        self.h = 0
        self.action_space_pre_embedding = torch.arange(self.actions_num, dtype=torch.float64).to(self.device).reshape(-1, 1)
        self.emiss_num = self.actions_num
        self.transition_matrix = None
        self._generate_emissions()
        self.action_space = self.emissions
        self.visitations = torch.zeros(self.states_num, self.actions_num, dtype=torch.float64).to(self.device)

    def get_dim(self):
        return 768

    def get_states_num(self):
        return self.states_num

    def _generate_emissions(self):
        
        # Create cache dir if needed
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Generate deterministic hash from actions and elements
        hasher = hashlib.sha256()
        hasher.update(str(self.actions_num).encode())
        for elem in self.unique_elements:
            hasher.update(elem.encode())
        cache_id = hasher.hexdigest()
        cache_path = os.path.join(self.cache_dir, f"emissions_{cache_id}.pkl")
    
        # Try loading from cache
        if os.path.exists(cache_path):
            if self.verbose:
                print("Loading emissions from cache")
            with open(cache_path, 'rb') as f:
                self.emissions = pickle.load(f)
            return
    
        # Generate if not cached
        if self.verbose:
            print("PREPROCESS: Generating emissions") 
        self.emissions = []
        for i in range(self.actions_num):
            text = self.unique_elements[i]
            if self.verbose:
                print(f"Generating emission for action {i}, text: {text}")
            feat = self.embedder.embed_text(text)
            self.emissions.append(feat)
        
        if self.verbose:
            print("Done generating.")
        self.emissions = torch.vstack(self.emissions)
        
        # Cache the emissions
        with open(cache_path, 'wb') as f:
            pickle.dump(self.emissions, f)

    #def _generate_emissions_legacy(self):
    #    # TODO: delete this
    #    if self.verbose:
    #        print("PREPROCESS: Generating emissions")
    #    self.emissions = []
    #    for i in range(self.actions_num):
    #        text = self.unique_elements[i]
    #        if self.verbose:
    #            print(f"Generating emission for action {i}, text: {text}")
    #        feat = self.embedder.embed_text(text)
    #        self.emissions.append(feat)
    #        
    #    if self.verbose:
    #        print("Done generating.")
    #    self.emissions = torch.vstack(self.emissions)
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

class CLIPEmbedder:
    """Embed text using CLIP model

    Args:
        text: Text to embed
        normalize: Whether to L2 normalize the embedding

    Returns:
        Text embedding
    """
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device  # Track model device

    def embed_text(self, text: str, normalize: bool = False) -> torch.Tensor:
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input = {k: v.to(self.device) for k, v in text_input.items()}
        embedding = self.model.get_text_features(**text_input).detach().double()

        if normalize:
            embedding = embedding / torch.norm(embedding, p=2)

        return embedding.view(1, -1)

class CLIPScorer(nn.Module):
    """Base class for CLIP-based scoring models"""
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder

    def score_prompt(self, x):
        raise NotImplementedError

class DotProductModel(CLIPScorer):
    def __init__(self, embedder, weight, bias=None):
        super().__init__(embedder)
        self.weight = weight.to(embedder.device)
        self.bias = bias.to(embedder.device) if bias else None
    
    def score_prompt(self, x):
        x_clip_embedding = self.embedder.embed_text(x)
        score = torch.mm(x_clip_embedding, self.weight.T)
        if self.bias:
            score += self.bias
        return score, x_clip_embedding

class ImageScorer(CLIPScorer):
    def __init__(self, embedder, cache_dir):
        super().__init__(embedder)
        self.generator = StableDiffusionGenerator("CompVis/stable-diffusion-v1-4", 
                                                MODELS_CACHE_DIR=cache_dir)
        
    def score_prompt(self, prompt):
        raise NotImplementedError

class RedImageScorer(ImageScorer):
    def __init__(self, embedder, cache_dir):
        super().__init__(embedder, cache_dir)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        red_image_size = tuple([1] + list(self.generator.image_size)[::-1])
        self.red_reference = torch.ones(red_image_size, device=device)
        self.red_reference[:, 1:, :, :] = 0
        
    def score_prompt(self, prompt):
        image, _ = self.generator.sample(prompt, raw=True)
        clip_embedding = self.embedder.embed_text(prompt)
        print('Should we do image[1] here?')
        import ipdb; ipdb.set_trace()
        
        image_norm = image / torch.norm(image)
        ref_norm = self.red_reference / torch.norm(self.red_reference)
        return torch.sum(image_norm * ref_norm), clip_embeddings

class AestheticsImageScorer(ImageScorer):
    def __init__(self, cache_dir, clip_model: CLIPModel, clip_processor: CLIPProcessor):
        super().__init__(cache_dir)
        # TODO: figure this one out
        #self.aesthetic_model = AestheticsModel('vit_14_weights.pth')
        #self.aesthetic_model = AestheticsModel('text_weights.pth')
        self.clip_processor = clip_processor
        self.clip_model = clip_model

    def score_prompt(self, prompt):
        image, _ = self.generator.sample(prompt, raw=True)
        print('Should we do image[1] here?')
        import ipdb; ipdb.set_trace()
        inputs = self.clip_processor(images=image, return_tensors="pt")
        clip_embeddings = self.clip_model.get_image_features(**inputs)
        return self.aesthetic_model(clip_embeddings), clip_embeddings

def load_aesthetics_embedding(weights_path='text_weights.pth'):
    """Load aesthetics model weights and bias
    
    Args:
        weights_path: Path to aesthetics weights file
        
    Returns:
        tuple: (weight tensor, bias tensor) both on appropriate device
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        state = torch.load(weights_path, map_location=device)
        weight = state['net.0.weight'].to(device).double()
        bias = state['net.0.bias'].to(device).double()
        norm = torch.norm(weight, p=2, dim=1, keepdim=True)
        return weight / norm , bias / norm
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find weights file: {weights_path}")

def setup_clip_model(cache_dir):
    """Initialize shared CLIP model"""
    model_id = "openai/clip-vit-large-patch14"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CLIPTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    return model, processor, tokenizer

def create_prompt(actions: List[int], env, prefix: str = 'A plate with ') -> str:
    """Create prompt from action sequence"""
    tokens = [env.unique_elements[int(action)] for action in actions]
    valid_tokens = [t for t in tokens if t != " "]
    return prefix + ", ".join(valid_tokens) if valid_tokens else prefix.rstrip()

def get_scorer_model(model_name: str, embedder, clip_model, clip_processor, cache_dir):
    """Initialize embedder and scoring model
    
    Args:
        embedder: CLIPEmbedder
        model_name: Scorer type ('art', 'aesthetics', 'aesthetics-image', 'red')
        clip_model: CLIP model, required for aesthetics-image
        clip_processor: CLIP processor, required for aesthetics-image 
        cache_dir: Cache directory for image scorers
    
    Returns:
        Tuple of (text_model, image_scorer), one will be None
    """
    if model_name == 'art':
        art_embedding = embedder.embed_text('art', normalize=True)
        return DotProductModel(embedder, art_embedding).eval()
        
    if model_name == 'aesthetics':
        aes_weight, aes_bias = load_aesthetics_embedding()
        return DotProductModel(embedder, aes_weight, bias=aes_bias).eval()
        
    if model_name == 'aesthetics-image':
        return  AestheticsImageScorer(embedder, cache_dir, clip_model, clip_processor)
        
    if model_name == 'red':
        return RedImageScorer(embedder, cache_dir)
        
    raise ValueError(f"Unknown model_name: {model_name}")

def make_theta_star(env, scorer_model):

    def theta_star(actions: List[int], verbose: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert len(actions) > 0
        prompt = create_prompt(actions, env)
        
        if verbose:
            print(prompt)

        score, clip_embedding = scorer_model.score_prompt(prompt)

        return (score, clip_embedding)
    
    return theta_star

