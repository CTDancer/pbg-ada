from collections import namedtuple
import os
import torch
import esm
from typing import List, Union, Optional
from protein_lm.modeling.scripts.train import compute_esm_embedding, compute_saprot_embedding, load_ckpt, make_esm_input_ids
from protein_lm.tokenizer.tokenizer import PTMTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import EsmTokenizer, EsmForMaskedLM

Output = namedtuple("output", ["logits", "hidden_states"])

class PTMMamba:
    def __init__(self, ckpt_path, device='cuda', ) -> None:
        self._tokenizer = PTMTokenizer()
        train_config = torch.load(ckpt_path, map_location='cpu')['config']
        self.use_esm = train_config.get('use_esm', True)
        self.use_saprot = train_config.get('use_saprot', False)
        assert not (self.use_esm and self.use_saprot), "Only one of use_esm and use_saprot can be True"
        saprot_path = train_config.get('saprot_path', "SaProt_650M_AF2/")
        if not os.path.exists(saprot_path):
            raise ValueError(f"Invalid saprot_path {saprot_path}")
        self._model = load_ckpt(ckpt_path, self.tokenizer, device)
        self._device = device
        self._model.to(device)
        self._model.eval()
        
        if self.use_esm:
            self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.esm_model.eval()
        elif self.use_saprot:
            self.saprot_tokenizer = EsmTokenizer.from_pretrained(saprot_path)
            self.saprot_model = EsmForMaskedLM.from_pretrained(saprot_path)
            self.esmfold_model = esm.pretrained.esmfold_v1().eval()

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def tokenizer(self) -> PTMTokenizer:
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        return self._device

    def infer(self, seq: str) -> Output:
        input_id = self.tokenizer(seq)
        input_ids = torch.tensor(input_id, device=self.device).unsqueeze(0)
        outputs = self._infer(input_ids)
        return outputs

    @torch.no_grad()
    def _infer(self, input_ids):
        if self.use_esm:
            esm_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
            embedding = compute_esm_embedding(
                self.tokenizer, self.esm_model, self.batch_converter, esm_input_ids
            )
        elif self.use_saprot:
            saprot_input_ids = make_esm_input_ids(input_ids, self.tokenizer)
            embedding = compute_saprot_embedding(
                self.saprot_tokenizer, self.saprot_model, self.esmfold_model, self.tokenizer, saprot_input_ids
            )
        else:
            embedding = None
        outputs = self.model(input_ids, embedding=embedding)
        return outputs

    def infer_batch(self, seqs: list) -> Output:
        input_ids = self.tokenizer(seqs)
        input_ids = pad_sequence(
            [torch.tensor(x) for x in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = torch.tensor(input_ids, device=self.device)
        outputs = self._infer(input_ids)
        return outputs

    def __call__(self, seq: Union[str, List]) -> Output:
        if isinstance(seq, str):
            return self.infer(seq)
        elif isinstance(seq, list):
            return self.infer_batch(seq)
        else:
            raise ValueError("Input must be a string or a list of strings, got {}".format(type(seq)))

if __name__ == "__main__":
    ckpt_path = "/home/tc415/ptm-mamba/ckpt/ptm-transformer/best.ckpt"
    mamba = PTMMamba(ckpt_path, device='cuda:0')
    seq = '<N-acetylmethionine>EAD<Phosphoserine>PAGPGAPEPLAEGAAAEFS<Phosphoserine>LLRRIKGKLFTWNILKTIALGQMLSLCICGTAITSQYLAERYKVNTPMLQSFINYCLLFLIYTVMLAFRSGSDNLLVILKRKWWKYILLGLADVEANYVIVRAYQYTTLTSVQLLDCFGIPVLMALSWFILHARYRVIHFIAVAVCLLGVGTMVGADILAGREDNSGSDVLIGDILVLLGASLYAISNVCEEYIVKKLSRQEFLGMVGLFGTIISGIQLLIVEYKDIASIHWDWKIALLFVAFALCMFCLYSFMPLVIKVTSATSVNLGILTADLYSLFVGLFLFGYKFSGLYILSFTVIMVGFILYCSTPTRTAEPAESSVPPVTSIGIDNLGLKLEENLQETH<Phosphoserine>AVL'
    output = mamba(seq)
    print(output.logits.shape)
    print(output.hidden_states.shape)
