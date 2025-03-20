import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple, Union

import torch
from pydantic import BaseModel

from python.bindings import executor as tllme
from python.logger import logger


@dataclass(slots=True, kw_only=True)
class GuidedDecodingParams:
    json: Optional[Union[str, BaseModel, dict]] = None
    regex: Optional[str] = None
    grammar: Optional[str] = None
    json_object: bool = False

    def _validate(self):
        num_guides = 0
        for field in fields(self):
            num_guides += bool(getattr(self, field.name))
        if num_guides > 1:
            raise ValueError(
                f"Only one guide can be used for a request, but got {num_guides}."
            )


class LogitsProcessor(ABC):

    @abstractmethod
    def __call__(self, req_id: int, logits: torch.Tensor,
                 token_ids: List[List[int]], stream_ptr: int,
                 client_id: Optional[int]) -> None:
        pass


class BatchedLogitsProcessor(ABC):

    @abstractmethod
    def __call__(self, req_ids: List[int], logits: List[torch.Tensor],
                 token_ids: List[List[List[int]]], stream_ptr: int,
                 client_ids: List[Optional[int]]) -> None:
        pass


@dataclass(slots=True, kw_only=True)
class AdditionalModelOutput:
    name: str
    gather_context: bool


@dataclass(slots=True, kw_only=True)
class SamplingParams:

    end_id: Optional[int] = None
    pad_id: Optional[int] = None
    max_tokens: int = 32
    max_new_tokens: Optional[int] = None

    bad: Optional[Union[str, List[str]]] = None
    bad_token_ids: Optional[List[int]] = None
    _bad_word_ids: Optional[List[List[int]]] = field(default=None,
                                                     init=False,
                                                     repr=False)
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: bool = False
    _stop_word_ids: Optional[List[List[int]]] = field(default=None,
                                                      init=False,
                                                      repr=False)

    embedding_bias: Optional[torch.Tensor] = None
    logits_processor: Optional[LogitsProcessor] = None
    apply_batched_logits_processor: bool = False

    n: int = 1
    best_of: Optional[int] = None
    use_beam_search: bool = False

    beam_width: int = 1
    num_return_sequences: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    top_p_min: Optional[float] = None
    top_p_reset_ids: Optional[int] = None
    top_p_decay: Optional[float] = None
    seed: Optional[int] = None
    random_seed: Optional[int] = None
    temperature: Optional[float] = None
    min_tokens: Optional[int] = None
    min_length: Optional[int] = None
    beam_search_diversity_rate: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[int] = None
    no_repeat_ngram_size: Optional[int] = None
    min_p: Optional[float] = None

    return_log_probs: bool = False
    return_context_logits: bool = False
    return_generation_logits: bool = False
    exclude_input_from_output: bool = True
    return_encoder_output: bool = False
    return_perf_metrics: bool = False
    additional_model_outputs: Optional[List[AdditionalModelOutput]] = None

    lookahead_config: Optional[tllme.LookaheadDecodingConfig] = None

    guided_decoding: Optional[GuidedDecodingParams] = None

    ignore_eos: bool = False
    detokenize: bool = True
    add_special_tokens: bool = True
    truncate_prompt_tokens: Optional[int] = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True

    def __post_init__(self):
        if self.pad_id is None:
            self.pad_id = self.end_id

        hf_style = self.beam_width > 1 or self.num_return_sequences
        openai_style = self.n > 1 or self.best_of or self.use_beam_search

        if hf_style and openai_style:
            ambiguous_params = {
                'beam_width': self.beam_width,
                'num_return_sequences': self.num_return_sequences,
                'n': self.n,
                'best_of': self.best_of,
                'use_beam_search': self.use_beam_search,
            }
            raise ValueError(
                'Got ambiguous parameters. Please specify either Hugging Face '
                'style parameters (beam_width or num_return_sequences) or '
                'OpenAI style parameters (n, best_of, or use_beam_search), '
                f'but not both: {ambiguous_params}. It is recommended to use '
                'OpenAI style parameters (n, best_of, use_beam_search).')

        if hf_style:
            logger.warning(
                "Please use 'n' and 'best_of' for the LLM API. The use of "
                "'beam_width' and 'num_return_sequences' will be deprecated "
                "in a future release.")
            self.n = self.beam_width
            self.best_of = self.num_return_sequences
            self.use_beam_search = self.beam_width > 1

        self.best_of = self.best_of or self.n

        if (not self.use_beam_search and self.n < self.best_of
                and not self.return_log_probs):
            logger.info(
                f"Enable 'return_log_probs' to trim the {self.n}-best among "
                f"{self.best_of} outputs under sampling decoding.")
            self.return_log_probs = True

        self._validate()

    def _validate(self):
        if self.best_of is not None:
            if self.best_of > 1 and self.best_of < self.n:
                raise ValueError(
                    f'In beam search, beam_width ({self.beam_width}) must be '
                    f'greater than or equal to num_return_sequences '
                    f'({self.num_return_sequences}).')

            if (self.best_of > 1 and self._greedy_decoding and
                    not os.environ.get('TLLM_ALLOW_N_GREEDY_DECODING', None)):
                raise ValueError(
                    f'Greedy decoding in the LLM API does not allow multiple '
                    f'returns. Please set to best_of=1, got best_of={self.best_of}. '
                    f'Please set to best_of=1 or set an environment variable '
                    f'TLLM_ALLOW_N_GREEDY_DECODING=1 to allow best_of > 1 '
                    f'under the greedy decoding.')

        if self.truncate_prompt_tokens is not None and self.truncate_prompt_tokens < 1:
            raise ValueError(
                f"truncate_prompt_tokens must be >= 1, got {self.truncate_prompt_tokens}"
            )

        if self.guided_decoding is not None:
            self.guided_decoding._validate()

    @property
    def _greedy_decoding(self) -> bool:
        return (not self.use_beam_search
                and (self.top_k is None or self.top_k == 1)
                and (self.top_p is None or self.top_p == 0.0))

    def _setup(self,
               tokenizer,
               add_special_tokens: bool = False) -> 'SamplingParams':
        if self.end_id is None:
            self.end_id = tokenizer.eos_token_id
            self.pad_id = tokenizer.pad_token_id
            if self.pad_id is None:
                self.pad_id = self.end_id

        if self.bad is not None:
            strs = [self.bad] if isinstance(self.bad, str) else self.bad
            self._bad_word_ids = [
                tokenizer.encode(s, add_special_tokens=add_special_tokens)
                for s in strs
            ]

        if self.stop is not None:
            strs = [self.stop] if isinstance(self.stop, str) else self.stop
            self._stop_word_ids = [
                tokenizer.encode(s, add_special_tokens=add_special_tokens)
                for s in strs
            ]

        return self

    def _get_bad_words(self) -> List[List[int]]:
        words = []
        if self.bad_token_ids:
            words = [[i] for i in self.bad_token_ids]

        if self.bad is None:
            return words
        else:
            if self._bad_word_ids is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.bad ({self.bad}) is not processed by tokenizer, "
                    "please call the setup method.")
            return words + self._bad_word_ids

    def _get_stop_words(self) -> List[List[int]]:
        words = []
        if self.stop_token_ids:
            words = [[i] for i in self.stop_token_ids]

        if self.stop is None:
            return words
        else:
            if self._stop_word_ids is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.stop ({self.stop}) is not processed by tokenizer, "
                    "please call the setup method.")
            return words + self._stop_word_ids

    def _get_stop_reasons_and_words(
            self) -> List[Tuple[Union[str, int], List[List[int]]]]:
        stop_reasons = []
        if self.stop_token_ids is not None:
            stop_reasons.extend(self.stop_token_ids)
        if self.stop is not None:
            if isinstance(self.stop, str):
                stop_reasons.append(self.stop)
            else:
                stop_reasons.extend(self.stop)
        stop_words = self._get_stop_words()
        return list(zip(stop_reasons, stop_words))

    def _get_sampling_config(self) -> tllme.SamplingConfig:
        fields = {
            f
            for f in dir(tllme.SamplingConfig) if not f.startswith('__')
        }
        unmatched_params = [
            'num_return_sequences', 'beam_width', 'n', 'best_of',
            'use_beam_search'
        ]
        llmapi_to_rt_param_map = {
            f: getattr(self, f)
            for f in fields if f not in unmatched_params
        }
        if self.use_beam_search:
            llmapi_to_rt_param_map['num_return_sequences'] = self.n
            llmapi_to_rt_param_map['beam_width'] = self.best_of
        else:
            llmapi_to_rt_param_map['num_return_sequences'] = self.best_of
            llmapi_to_rt_param_map['beam_width'] = 1

        return tllme.SamplingConfig(**llmapi_to_rt_param_map)

    def _get_output_config(self) -> tllme.OutputConfig:
        fields = [f for f in dir(tllme.OutputConfig) if not f.startswith('__')]
        return tllme.OutputConfig(**{f: getattr(self, f) for f in fields})

    def _get_guided_decoding_params(self) -> tllme.GuidedDecodingParams:
        if self.guided_decoding is None:
            return None

        if self.guided_decoding.json_object:
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.JSON)
        elif self.guided_decoding.json is not None:
            json_schema = self.guided_decoding.json
            if isinstance(json, BaseModel):
                json_schema = json_schema.dict()
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.JSON, json_schema)

        if self.guided_decoding.regex is not None:
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.REGEX,
                self.guided_decoding.regex)

        if self.guided_decoding.grammar is not None:
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.GRAMMAR,
                self.guided_decoding.grammar)

        return None
