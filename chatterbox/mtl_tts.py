from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, S3_TOKEN_RATE, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"

# Checkpoint selection. The defaults reproduce the original behaviour exactly.
DEFAULT_T3_MODEL = "t3_mtl23ls_v2.safetensors"
T3_MODELS = {
    "v2": "t3_mtl23ls_v2.safetensors",
    "v3": "t3_mtl23ls_v3.safetensors",
}

DEFAULT_S3GEN_MODEL = "s3gen.pt"
S3GEN_MODELS = {
    "v1": "s3gen.pt",
    "v3": "s3gen_v3.pt",
}


def _resolve(alias, table, default, kind):
    if alias is None:
        return default
    if alias in table:
        return table[alias]
    if alias.endswith((".safetensors", ".pt")):
        return alias
    raise ValueError(
        f"Unknown {kind} '{alias}'. Choose from {sorted(table)} or pass a filename."
    )


# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR
    is_t3_v3 = False  # set by from_local() when the v3 T3 checkpoint is loaded

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device, t3_model=None, s3gen_model=None) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)
        t3_file = _resolve(t3_model, T3_MODELS, DEFAULT_T3_MODEL, "t3_model")
        s3gen_file = _resolve(s3gen_model, S3GEN_MODELS, DEFAULT_S3GEN_MODEL, "s3gen_model")

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / t3_file)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        # v3 was trained without the alignment analyzer; leaving it on can force
        # a premature EOS and truncate the audio.
        t3.use_alignment_analyzer = t3_file != T3_MODELS["v3"]
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen_state = torch.load(
            ckpt_dir / s3gen_file,
            weights_only=True,
            map_location=torch.device(device)
        )
        # s3gen_v3 ships without the tokenizer._mel_filters / tokenizer.window
        # buffers; both are recomputed deterministically in __init__.
        s3gen.load_state_dict(s3gen_state, strict=s3gen_file == DEFAULT_S3GEN_MODEL)

        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        model = cls(t3, s3gen, ve, tokenizer, device, conds=conds)
        model.is_t3_v3 = t3_file == T3_MODELS["v3"]
        return model

    @classmethod
    def from_pretrained(cls, device: torch.device, t3_model=None, s3gen_model=None) -> 'ChatterboxMultilingualTTS':
        t3_file = _resolve(t3_model, T3_MODELS, DEFAULT_T3_MODEL, "t3_model")
        s3gen_file = _resolve(s3gen_model, S3GEN_MODELS, DEFAULT_S3GEN_MODEL, "s3gen_model")
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=["ve.pt", t3_file, s3gen_file, "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device, t3_model=t3_file, s3gen_model=s3gen_file)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=None,
        min_p=0.05,
        top_p=1.0,
    ):
        if repetition_penalty is None:
            repetition_penalty = 1.2 if self.is_t3_v3 else 2.0

        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            if self.is_t3_v3:
                # Drop the final speech token's audio: it is emitted just before
                # EOS with degraded attention and decodes to ~40 ms of noise.
                st_len = max(1, int(speech_tokens.shape[-1]) - 1)
                wav = wav[: st_len * (S3GEN_SR // S3_TOKEN_RATE)]
            # watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(wav).unsqueeze(0)







from pathlib import Path
import torch


class CachedVoiceChatterboxTTS(ChatterboxMultilingualTTS):
    """
    Extension of ChatterboxMultilingualTTS that supports
    one-time voice cloning and persistent caching of conditionals.
    """

    def clone_voice_once(
        self,
        audio_prompt_path: str | Path,
        cache_path: str | Path,
        exaggeration: float = 0.5,
        overwrite: bool = False,
    ) -> Conditionals:
        """
        Clone voice from audio once and save conditionals to disk.

        Args:
            audio_prompt_path: Reference audio path
            cache_path: Where to store conditionals (.pt)
            exaggeration: Emotion exaggeration factor
            overwrite: Recompute even if cache exists
        """
        cache_path = Path(cache_path)

        if cache_path.exists() and not overwrite:
            raise FileExistsError(
                f"Conditionals already exist at {cache_path}. "
                f"Use overwrite=True to recompute."
            )

        # Run expensive voice cloning ONCE
        self.prepare_conditionals(
            wav_fpath=audio_prompt_path,
            exaggeration=exaggeration,
        )

        # Save conditionals
        self.conds.save(cache_path)

        return self.conds

    def load_cloned_voice(
        self,
        cache_path: str | Path,
    ) -> None:
        """
        Load cached voice conditionals from disk.
        """
        cache_path = Path(cache_path)

        if not cache_path.exists():
            raise FileNotFoundError(f"No cached conditionals found at {cache_path}")

        self.conds = Conditionals.load(
            cache_path,
            map_location=self.device,
        ).to(self.device)

    def generate_with_cached_voice(
        self,
        text: str,
        language_id: str,
        exaggeration: float = 0.5,
        **gen_kwargs,
    ):
        """
        Generate speech using already-loaded cached voice.
        """
        assert self.conds is not None, (
            "No cached voice loaded. "
            "Call load_cloned_voice() first."
        )

        return self.generate(
            text=text,
            language_id=language_id,
            audio_prompt_path=None,  # IMPORTANT: skip re-cloning
            exaggeration=exaggeration,
            **gen_kwargs,
        )
