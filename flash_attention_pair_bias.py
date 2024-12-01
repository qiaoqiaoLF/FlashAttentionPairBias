"""
Fused Attention with Pair Bias Support
======================================

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf),
modified to support pair bias with gradients.

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

Modified by: Lifeng Qiao 2024
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr, BIAS_block_ptr, #
                    start_m, qk_scale, bias_scale, #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # range of values handled by this stage

    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    BIAS_block_ptr = tl.advance(BIAS_block_ptr, (0, lo))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        # -- compute qk + bias ----
        bias = tl.load(BIAS_block_ptr)

        qk = qk * qk_scale + bias * bias_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)

        p = p.to(tl.bfloat16)
        
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        BIAS_block_ptr = tl.advance(BIAS_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


configs_fwd = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in [3, 4, 5, 6, 7]\
    for w in [4, 8]\
]


def keep_fwd(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    if BLOCK_M % BLOCK_N != 0:
        return False
    return True


@triton.autotune(list(filter(keep_fwd, configs_fwd)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, BIAS, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_biasz, stride_biash, stride_biasm, stride_biasn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    bias_offset = off_z.to(tl.int64) * stride_biasz + off_h.to(tl.int64) * stride_biash
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    BIAS_block_ptr = tl.make_block_ptr(
        base=BIAS + bias_offset,
        shape=(N_CTX, N_CTX),
        strides=(stride_biasm, stride_biasn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    bias_scale = 1.44269504 
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, BIAS_block_ptr, #
                                    start_m, qk_scale, bias_scale, #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    offs_m, offs_n, N_CTX  #
                                    )
    
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    

configs_bwd_pre = [
    triton.Config({'BLOCK_M': BM}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for s in [3, 4, 5, 6, 7]\
    for w in [4, 8]\
]

def keep_bwd_pre(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    if BLOCK_M  < 128  and conf.num_warps == 8:
        return False
    return True

@triton.autotune(list(filter(keep_bwd_pre, configs_bwd_pre)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         HEAD_DIM: tl.constexpr, #
                         BLOCK_M: tl.constexpr,  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


configs_bwd_dkdv = [
    triton.Config({'BLOCK_M1': BM1, 'BLOCK_N1': BN1}, num_stages=s, num_warps=w) \
    for BM1 in [32, 64]\
    for BN1 in [64, 128]\
    for s in [3, 4, 5, 6, 7]\
    for w in [4, 8]\
]


def keep_bwd_dkdv(conf):
    BLOCK_M1 = conf.kwargs["BLOCK_M1"]
    BLOCK_N1 = conf.kwargs["BLOCK_N1"]
    if BLOCK_M1 * BLOCK_N1 < 128 * 128 and conf.num_warps == 8:
        return False
    if BLOCK_N1 % BLOCK_M1 != 0:
        return False
    return True


@triton.autotune(list(filter(keep_bwd_dkdv, configs_bwd_dkdv)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_dkdv(Q, K, V, BIAS, sm_scale,  #
                   DK, DV ,DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_z, stride_h, stride_tok, stride_d,  #
                   bias_stride_z, bias_stride_h, bias_stride_m, bias_stride_n,  #
                   H, N_CTX, HEAD_DIM: tl.constexpr,  #
                   BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  ):
    
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    adj_bias = (bias_stride_h * (bhid % H) + bias_stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    BIAS += adj_bias
    DO += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    
    start_n = pid * BLOCK_N1
    start_m = 0
    num_steps = (N_CTX) // BLOCK_M1
    
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)

    # TO BE LOOOPED OVER
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    biasT_ptrs = BIAS + offs_m[None, :] * bias_stride_m + offs_n[:, None] * bias_stride_n
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    
    
    # NOT TO BE LOOPED OVER
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        biasT = tl.load(biasT_ptrs)
        pT = tl.math.exp2(qkT + biasT  - m[None, :])

        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.bfloat16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.bfloat16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        biasT_ptrs += step_m * bias_stride_m
        
    # Write back dV.
    tl.store(dv_ptrs, dv)
    # Write back dK.
    dk *= sm_scale
    tl.store(dk_ptrs, dk)

configs_bwd_dqdbias = [
    triton.Config({'BLOCK_M2': BM2, 'BLOCK_N2': BN2}, num_stages=s, num_warps=w) \
    for BM2 in [64, 128]\
    for BN2 in [32, 64]\
    for s in [3, 4, 5, 6, 7]\
    for w in [4, 8]\
]

def keep_bwd_dqdbias(conf):
    BLOCK_M2 = conf.kwargs["BLOCK_M2"]
    BLOCK_N2 = conf.kwargs["BLOCK_N2"]
    if BLOCK_M2 * BLOCK_N2 < 128 * 128 and conf.num_warps == 8:
        return False
    if BLOCK_M2 % BLOCK_N2 != 0:
        return False
    return True


@triton.autotune(list(filter(keep_bwd_dqdbias, configs_bwd_dqdbias)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_dqdbias(Q, K, V, BIAS,#
                 DQ, DBIAS, DO, 
                 M, D,
                 # shared by Q/K/V/DO.
                 stride_z, stride_h, stride_tok, stride_d,  #
                 bias_stride_z, bias_stride_h, bias_stride_m, bias_stride_n,  #
                 H, N_CTX, HEAD_DIM: tl.constexpr,
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  ):
    
    LN2 = 0.6931471824645996  # = ln(2)
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    adj_bias = (bias_stride_h * (bhid % H) + bias_stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)
    
    Q += adj
    K += adj
    V += adj
    BIAS += adj_bias
    DO += adj
    DQ += adj
    DBIAS += adj_bias
    M += off_chz
    D += off_chz
    
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2
    
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # TO BE LOOOPED OVER
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    bias_ptrs = BIAS + offs_m[:, None] * bias_stride_m + offs_n[None, :] * bias_stride_n
    dbias_ptrs = DBIAS + offs_m[:, None] * bias_stride_m + offs_n[None, :] * bias_stride_n
    
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    
    # NOT TO BE LOOPED OVER
    m = tl.load(M + offs_m)
    m = m[:, None]
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        bias = tl.load(bias_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk + bias - m)
        
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        
        # dbias = ds
        tl.store(dbias_ptrs, ds)
        
        ds = ds.to(tl.bfloat16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
        bias_ptrs += step_n * bias_stride_n
        dbias_ptrs += step_n * bias_stride_n
        
    dq *= LN2
    tl.store(dq_ptrs, dq)

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_M"]), BATCH * N_HEAD, 1)
        M = torch.empty((BATCH, N_HEAD, N_CTX), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, bias, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3), #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            BATCH, N_HEAD,  #
            N_CTX=N_CTX,  #
            HEAD_DIM=HEAD_DIM_K)

        ctx.save_for_backward(q, k, v, bias, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v , bias, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbias = torch.empty_like(bias)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        arg_bias = bias
        arg_bias = arg_bias * RCP_LN2
        pre_grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_M"]), BATCH * N_HEAD, 1)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX=N_CTX,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
        )
        
        grid_kv = lambda args: (triton.cdiv(N_CTX, args["BLOCK_N1"]), 1, BATCH * N_HEAD)
        _attn_bwd_dkdv[grid_kv](
            q, arg_k, v, arg_bias, ctx.sm_scale,  #
            dk, dv, do,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),  #
            N_HEAD, N_CTX=N_CTX, HEAD_DIM=ctx.HEAD_DIM, )
        
        grid_dqdbias = lambda args: (triton.cdiv(N_CTX, args["BLOCK_M2"]), 1, BATCH * N_HEAD)
        _attn_bwd_dqdbias[grid_dqdbias](
            q, arg_k, v, arg_bias,  #
            dq, dbias, do,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),  #
            N_HEAD, N_CTX=N_CTX, HEAD_DIM=ctx.HEAD_DIM, )
             

        return dq, dk, dv, dbias, None, None


attention = _attention.apply

def test_op(Z, H, N_CTX, HEAD_DIM):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.bfloat16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.bfloat16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.bfloat16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    bias = (torch.empty((Z, H, N_CTX, N_CTX), dtype=torch.bfloat16, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()) 
    sm_scale = 1 / (HEAD_DIM ** 0.5)
    dout = torch.randn_like(q)
    # # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = p + bias

    p = torch.softmax(p.float(), dim=-1).to(torch.bfloat16)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    ref_dbias, bias.grad = bias.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, bias, sm_scale).to(torch.bfloat16)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    tri_dbias, bias.grad = bias.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0.0)
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0.0)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0.0)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0.0)
    assert torch.allclose(ref_dbias, tri_dbias, atol=1e-2, rtol=0.0)



if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    test_op(1, 2, 1024, 64)
