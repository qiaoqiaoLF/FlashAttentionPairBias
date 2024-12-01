# FlashAttentionPairBias

## Overview

**FlashAttentionPairBias** is an enhanced Triton implementation of the **Flash Attention v2** algorithm, based on the paper [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), originally developed by Tri Dao. This modified version supports **pair bias with gradients**, enabling improved performance in specific attention-related tasks where pairwise dependencies are critical, such as in AlphaFold. Currently, only the non-causal version is supported. However, implementing the causal version is straightforward and can be easily adapted by referring to existing implementations of causal attention.

The underlying approach builds on the Flash Attention v2 paper (https://arxiv.org/abs/2307.08691) and its optimizations for fast attention computation. This modification incorporates a pair bias mechanism, making it suitable for advanced use cases requiring gradient computation over pairwise interactions.

This implementation is developed based on the [**Triton version by the OpenAI Team**](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py).

Note: The autotuning of Triton may be time-consuming. You can set the environment variable TRITON_PRINT_AUTOTUNING to "1", so that Triton will print a message to stdout after autotuning each kernel, including the time spent autotuning and the best configuration.
